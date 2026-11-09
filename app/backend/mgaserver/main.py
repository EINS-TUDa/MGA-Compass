from contextlib import asynccontextmanager
import logging

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

import mga_format
from mga_compass import Constraints, Alpha, AlphaSumError, Breakpoint, SolverError, navigate, interpolate, describe_changes, linear_lower_bounds, upper_envelope, configure_solver_logging
from mga_compass.near_optimal import category_styles, display_points, display_units, plot_data

from .api_models import ConstraintRow, ConstraintsPayload, BreakpointResult, NavigateResponse, ConstraintChangeItem, DimensionRange, DimensionEntry, InitResponse, InitPlotResponse, PlotRequest, PlotResponse, LowerBoundRequest, LowerBoundResponse, LowerBoundPointResult
from .config import settings

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

configure_solver_logging(settings.solver_log_path)

nos: mga_format.NearOptimalSet
# points: one row per point (index: point id), one column per PoI plus the cost
# column, in the manifest's display units. The whole API speaks display units:
# what the frontend shows, it also sends back.
points: pd.DataFrame = pd.DataFrame()
obj_label: str = ""
# Empty: /lower_bound passes it to linear_lower_bounds, but the cost lower bound
# is not enabled yet, although the set now always carries duals (nos.duals).
duals: pd.DataFrame = pd.DataFrame()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global nos, points, obj_label
    if settings.dataset_url:
        logger.info("fetching near-optimal set %s into %s", settings.dataset_url, settings.data_dir)
        nos = mga_format.fetch(settings.dataset_url, settings.data_dir)
    else:
        # Verified against the manifest's checksums before anything is read.
        nos = mga_format.read(settings.data_dir)
    points = display_points(nos)
    obj_label = nos.manifest.cost.name
    # Outputs stay on disk: /plot_data reads only the points a request mixes.
    logger.info("serving %d points, slack levels %s (navigating the largest)", len(points), nos.manifest.slacks)
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
    optimal_point = points.loc[points[obj_label].idxmin()]
    info = {poi.name: poi.description for poi in nos.manifest.pois}
    units = display_units(nos)
    return InitResponse(
        obj_label=obj_label,
        dimensions={
            col: DimensionEntry(
                range=DimensionRange(min=float(points[col].min()), max=float(points[col].max())),
                info=info.get(col, ""),
                unit=units.get(col, ""),
            )
            for col in points.columns
        },
        point=optimal_point.to_dict(),
    )


@app.get("/init_plot", response_model=InitPlotResponse)
@limiter.limit("60/minute")
def get_init_plot(request: Request) -> InitPlotResponse:
    labels, colors = category_styles(nos)
    return InitPlotResponse(
        datasets=list(nos.manifest.plots.keys()), obj_label=obj_label, labels=labels, colors=colors,
    )


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
    try:
        logger.info("navigate: calling navigate()")
        output_constraints, point = navigate(points, input_constraints, obj_label, settings.solver_name, settings.solver_options)
        logger.info("navigate: calling interpolate()")
        logger.info("interpolate point_start:\n%s", input_constraints["value"].to_string())
        logger.info("interpolate point_end:\n%s", point.to_string())
        logger.info("constraints:\n%s", input_constraints[["value", "direction", "delta"]].to_string())
        breakpoints = interpolate(points, input_constraints["value"], point, obj_label, solver_name=settings.solver_name, solver_options=settings.solver_options)
    except (SolverError, AlphaSumError) as e:
        # An HTTPException, unlike an unhandled error, still passes through the
        # CORS middleware, so the frontend sees this message, not a network error.
        logger.error("navigate: %s", e)
        raise HTTPException(status_code=500, detail=f"The solver failed: {e}")
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
    if payload.name not in nos.manifest.plots:
        raise HTTPException(status_code=404, detail=f"Plot '{payload.name}' not found")

    plot = nos.manifest.plots[payload.name]
    alphas = [Alpha(pd.Series({int(k): v for k, v in item.alpha.items()})) for item in payload.breakpoints]
    result = plot_data(settings.data_dir, nos, payload.name, alphas, display=True)
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
    bounds = linear_lower_bounds(points, breakpoint_objs, duals, obj_label)
    envelope = upper_envelope(bounds)
    return LowerBoundResponse(points=[LowerBoundPointResult(beta=p.beta, value=p.value) for p in envelope])


