from contextlib import asynccontextmanager
import logging
import pickle

import pandas as pd
import xarray as xr
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from mga_compass import Constraints, Alpha, Breakpoint, navigate, interpolate, get_plot_data, describe_changes, linear_lower_bounds, upper_envelope, configure_solver_logging

from .api_models import ConstraintRow, ConstraintsPayload, BreakpointResult, NavigateResponse, ConstraintChangeItem, DimensionRange, DimensionEntry, InitResponse, InitPlotResponse, PlotRequest, PlotResponse, LowerBoundRequest, LowerBoundResponse, LowerBoundPointResult
from .config import settings, load_manifest, Manifest

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

configure_solver_logging(settings.solver_log_path)

points: pd.DataFrame = pd.DataFrame()
duals: pd.DataFrame = pd.DataFrame()
outer_approximation: pd.DataFrame
output_datasets: dict[str, xr.DataArray]
manifest: Manifest


@asynccontextmanager
async def lifespan(app: FastAPI):
    global points, duals, outer_approximation, output_datasets, manifest
    manifest = load_manifest(settings.manifest_path)


    # points pickle: DataFrame, one row per MGA solution point.
    #   Index  : integer (0-based)
    #   Columns: one per decision variable (e.g. wind, solar, …) + objective (e.g. TOTEX)
    #   Example:
    #     wind  solar  hydrogen  battery  TOTEX
    #     10    20     45        25       35
    points = pd.read_pickle(settings.data_path)

    # duals pickle: DataFrame, one row per point (same integer index as points).
    #   Columns: one per non-objective decision variable containing the dual value
    #            (shadow price) of the capacity constraint at that solution point.
    #   Example:
    #     wind   solar  hydrogen  battery
    #     0.12   0.08   0.05      0.03
    # duals = pd.read_pickle(settings.duals_path)

    # outer_approximation pickle: DataFrame, one row per linear constraint.
    #   Columns: one per decision variable (coefficient) + "direction" + "RHS".
    #   "direction": str, one of ">=", "<=", "=="
    #   "RHS"      : float, right-hand side value
    #   Example:
    #     wind  solar  TOTEX  direction  RHS
    #     1.0   0.5    0.0    >=         10.0
    #     0.0   1.0   -1.0    <=          0.0
    # outer_approximation = pd.read_pickle(settings.outer_approximation_path)

    output_datasets = {name: pickle.loads(ds.path.read_bytes()) for name, ds in manifest.output_datasets.items()}
    yield


limiter = Limiter(key_func=get_remote_address)

app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request, call_next):
    logger.info("→ %s %s", request.method, request.url.path)
    response = await call_next(request)
    logger.info("← %s %s %s", request.method, request.url.path, response.status_code)
    return response


@app.get("/init", response_model=InitResponse)
@limiter.limit("60/minute")
def get_init(request: Request) -> InitResponse:
    optimal_point = points.loc[points[manifest.obj_label].idxmin()]
    return InitResponse(
        obj_label=manifest.obj_label,
        dimensions={
            col: DimensionEntry(
                range=DimensionRange(min=float(points[col].min()), max=float(points[col].max())),
                info=manifest.dimension_info.get(col, ""),
                unit=manifest.dimension_unit.get(col, ""),
            )
            for col in points.columns
        },
        point=optimal_point.to_dict(),
    )


@app.get("/init_plot", response_model=InitPlotResponse)
@limiter.limit("60/minute")
def get_init_plot(request: Request) -> InitPlotResponse:
    return InitPlotResponse(datasets=list(manifest.plots.keys()), obj_label=manifest.obj_label)


@app.post("/navigate", response_model=NavigateResponse)
@limiter.limit("20/minute;60/5minutes")
def run_navigation(request: Request, payload: ConstraintsPayload) -> NavigateResponse:
    logger.info("navigate: building constraints")
    input_constraints = Constraints(pd.DataFrame(
        {label: row.model_dump() for label, row in payload.constraints.items()}
    ).T.astype({"delta": float, "value": float}))
    try:
        logger.info("navigate: validating constraints")
        input_constraints.validate(points)
    except ValueError as e:
        # 422 Unprocessable Entity: the request is syntactically valid JSON but the
        # constraint values are semantically invalid (e.g. out of range, unknown dimension,
        # negative delta). FastAPI uses 422 for its own validation errors, so we match
        # that convention here rather than using 400 Bad Request.
        raise HTTPException(status_code=422, detail=str(e))
    logger.info("navigate: calling navigate()")
    output_constraints, point = navigate(points, input_constraints, manifest.obj_label, settings.solver_name, settings.solver_options)
    logger.info("navigate: calling interpolate()")
    logger.info("interpolate point_start:\n%s", input_constraints["value"].to_string())
    logger.info("interpolate point_end:\n%s", point.to_string())
    logger.info("constraints:\n%s", input_constraints[["value", "direction", "delta"]].to_string())
    breakpoints = interpolate(points, input_constraints["value"], point, manifest.obj_label, solver_name=settings.solver_name, solver_options=settings.solver_options)
    logger.info("navigate: building response")
    constraints_out = {
        label: ConstraintRow(**row.to_dict())
        for label, row in output_constraints.iterrows()
    }
    breakpoints_out = [
        BreakpointResult(beta=bp.beta, alpha={str(k): v for k, v in bp.alpha.to_dict().items()}, point=bp.point.to_dict())
        for bp in breakpoints
    ]
    changes_out = [ConstraintChangeItem(code=c.code, message=c.message, values=c.values, dim=c.dim) for c in describe_changes(input_constraints, output_constraints)]
    return NavigateResponse(constraints=constraints_out, breakpoints=breakpoints_out, changes=changes_out, point=point.to_dict())


@app.post("/plot_data", response_model=PlotResponse)
@limiter.limit("30/minute")
def get_plot_endpoint(request: Request, payload: PlotRequest) -> PlotResponse:
    if payload.name not in manifest.plots:
        raise HTTPException(status_code=404, detail=f"Plot '{payload.name}' not found")

    plot = manifest.plots[payload.name]
    alphas = [Alpha(pd.Series({int(k): v for k, v in item.alpha.items()})) for item in payload.breakpoints]
    da = output_datasets[plot.dataset]
    result = get_plot_data(alphas, plot, da)
    return PlotResponse(type=plot.type, betas=[item.beta for item in payload.breakpoints], x_dim=result["x_dim"], data=result["data"])


@app.post("/lower_bound", response_model=LowerBoundResponse)
@limiter.limit("60/minute")
def get_lower_bound(request: Request, payload: LowerBoundRequest) -> LowerBoundResponse:
    breakpoint_objs = [
        Breakpoint(
            beta=item.beta,
            alpha=Alpha(pd.Series({int(k): v for k, v in item.alpha.items()})),
            point=pd.Series(item.point),
        )
        for item in payload.breakpoints
    ]
    bounds = linear_lower_bounds(points, breakpoint_objs, duals, manifest.obj_label)
    envelope = upper_envelope(bounds)
    return LowerBoundResponse(points=[LowerBoundPointResult(beta=p.beta, value=p.value) for p in envelope])


