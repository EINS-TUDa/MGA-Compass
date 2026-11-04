#: Feasibility tolerance of every LP: HiGHS' own default. Needs data whose
#: dimensions span similar ranges (see the dataset build scripts).
FEASIBILITY_TOLERANCE = 1e-7

_SOLVER_SILENCE: dict[str, dict] = {
    "highs": {"output_flag": False},
}

#: HiGHS' simplex (its default) gives up -- status "unknown" -- on some of
#: point_2_alpha's and navigate's LPs at points where many dimensions sit at the
#: edge of their range, with or without presolve; its interior-point method, which
#: ends in a crossover to a vertex, solves them all and is as fast.
_SOLVER_METHOD: dict[str, dict] = {
    "highs": {"solver": "ipm"},
}

_SOLVER_FEASIBILITY_TOLERANCE_KEY: dict[str, str] = {
    "highs": "primal_feasibility_tolerance",
}


def default_solver_options(solver_name: str, feasibility_tolerance: float = FEASIBILITY_TOLERANCE) -> dict:
    opts = {**_SOLVER_SILENCE.get(solver_name, {}), **_SOLVER_METHOD.get(solver_name, {})}
    tol_key = _SOLVER_FEASIBILITY_TOLERANCE_KEY.get(solver_name)
    if tol_key:
        opts[tol_key] = feasibility_tolerance
    return opts
