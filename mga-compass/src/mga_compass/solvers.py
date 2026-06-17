_SOLVER_SILENCE: dict[str, dict] = {
    "highs": {"output_flag": False},
}

_SOLVER_FEASIBILITY_TOLERANCE_KEY: dict[str, str] = {
    "highs": "primal_feasibility_tolerance",
}


def default_solver_options(solver_name: str, feasibility_tolerance: float = 1e-4) -> dict:
    opts = dict(_SOLVER_SILENCE.get(solver_name, {}))
    tol_key = _SOLVER_FEASIBILITY_TOLERANCE_KEY.get(solver_name)
    if tol_key:
        opts[tol_key] = feasibility_tolerance
    return opts
