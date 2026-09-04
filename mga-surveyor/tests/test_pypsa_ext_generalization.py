"""Broader validation of generate_mga_files: the degenerate-dual
(_dual_scale) fix, VertexEnumMethod's halfspace normalization, and
use_objective_cuts were all originally debugged and validated only
against pypsa.examples.model_energy() solved with Gurobi -- these two
tests re-run the same convergence + independent-ground-truth methodology
against a network other than model_energy, and against model_energy
again but under solver_name="highs", each changing exactly one variable
from the original validation so a future failure points at whether it's
network- or solver-specific rather than conflating the two.

Cross-validation method: generate_mga_files is driven to convergence
while spying on the live VertexEnumMethod it builds internally (there's
no other way to get at it -- generate_mga_files only writes .nc files),
then an independent CddMethod is bootstrapped with that same method's own
final (inner_vertices, outer_halfspaces) and checked for agreement.
CddMethod uses cddlib's Double Description algorithm, an entirely
different library/algorithm from VertexEnumMethod's own scipy-Qhull-plus-
incremental-cut construction -- agreement is a real independent
correctness check, not two runs of the same code.

Marked slow: real PyPSA network solves, up to 100 iterations each, plus
an independent CddMethod cross-check.
"""
import pypsa
import pytest
from scipy.spatial import ConvexHull

import mga_surveyor.pypsa_ext as pypsa_ext_mod
from mga_surveyor import CddMethod, generate_mga_files

pytestmark = pytest.mark.slow

TOL_PERCENT = 4.0
TOL_CUBE = TOL_PERCENT / 100 * 2  # generate_mga_files' own normalize=True conversion


def _converge_and_cross_validate(monkeypatch, n, dimensions, output_dir, norm, **generate_kwargs):
    """Runs generate_mga_files (one slack level) to convergence, capturing
    the live VertexEnumMethod via a spy on pypsa_ext's own `drive`
    reference (generate_mga_files calls it once per slack level; patching
    pypsa_ext_mod.drive specifically, not explore.drive, since that's the
    name pypsa_ext.py actually calls -- `from .explore import drive`
    binds its own local reference at import time). Returns
    (method, method_cdd) for the caller to compare.
    """
    captured = {}
    orig_drive = pypsa_ext_mod.drive

    def spy_drive(method, *args, **kwargs):
        result = orig_drive(method, *args, **kwargs)
        captured["method"] = method
        return result

    monkeypatch.setattr(pypsa_ext_mod, "drive", spy_drive)
    generate_mga_files(n, dimensions, output_dir, norm=norm, **generate_kwargs)
    method = captured["method"]

    method_cdd = CddMethod(k=len(dimensions), norm=norm)
    method_cdd.bootstrap(list(method.inner_vertices), method.outer_halfspaces)
    return method, method_cdd


def test_converges_and_matches_cdd_on_ac_dc_meshed(tmp_path, monkeypatch):
    """Network-generality check: pypsa.examples.ac_dc_meshed() is a
    multi-bus meshed AC/DC topology (3 zones, wind+gas generation, DC
    links) -- structurally nothing like model_energy's single-node
    wind/solar/battery/hydrogen setup, so it stresses different
    degeneracy patterns. Still Gurobi + use_objective_cuts' own default
    (True), isolating the network as the only changed variable.
    """
    n = pypsa.examples.ac_dc_meshed()
    n.optimize(solver_name="gurobi", solver_options={"env": {"OutputFlag": 0}}, include_objective_constant=True)
    dimensions = {
        "manchester_wind": {"Generator": {"p_nom": {"Manchester Wind": 1}}},
        "norway_wind": {"Generator": {"p_nom": {"Norway Wind": 1}}},
        "frankfurt_wind": {"Generator": {"p_nom": {"Frankfurt Wind": 1}}},
    }

    method, method_cdd = _converge_and_cross_validate(
        monkeypatch, n, dimensions, tmp_path, norm="linf", slack_levels=[0.05], solver_name="gurobi",
        tol=TOL_PERCENT, normalize=True, max_iter=100, max_parallel=2, verbose=False,
        solver_options={"env": {"OutputFlag": 0}},
    )

    assert method.max_gap() < TOL_CUBE
    assert method_cdd.max_gap() == pytest.approx(method.max_gap(), abs=1e-6)

    ve_inner_vol = ConvexHull(method.inner_vertices).volume
    cdd_inner_vol = ConvexHull(method_cdd.inner_vertices).volume
    assert ve_inner_vol == pytest.approx(cdd_inner_vol, rel=1e-4)


def test_converges_and_matches_cdd_on_model_energy_with_highs(tmp_path, monkeypatch):
    """Solver-generality check: model_energy again, but solver_name="highs"
    instead of gurobi -- the degenerate-dual fix's own numerical-
    conditioning argument (rescaling to move a dual's magnitude away from
    the solver's own OptimalityTol floor) is solver-agnostic in principle,
    but every original diagnostic that confirmed it empirically ran on
    Gurobi. Isolates the solver as the only changed variable.
    """
    n = pypsa.examples.model_energy()
    n.snapshots = n.snapshots[::3]  # crude time-resolution reduction for speed, same as generate_dataset.py
    n.generators.at["load shedding", "marginal_cost"] = 20000
    n.optimize(solver_name="highs", solver_options={"output_flag": False}, include_objective_constant=True)
    dimensions = {
        "wind": {"Generator": {"p_nom": {"wind": 1}}},
        "solar": {"Generator": {"p_nom": {"solar": 1}}},
        "battery": {"StorageUnit": {"p_nom": {"battery storage": 1}}},
        "hydrogen": {"Store": {"e_nom": {"hydrogen storage": 1}}},
    }

    method, method_cdd = _converge_and_cross_validate(
        monkeypatch, n, dimensions, tmp_path, norm="linf", slack_levels=[0.05], solver_name="highs",
        tol=TOL_PERCENT, normalize=True, max_iter=100, max_parallel=2, verbose=False,
        solver_options={"output_flag": False},
    )

    assert method.max_gap() < TOL_CUBE
    assert method_cdd.max_gap() == pytest.approx(method.max_gap(), abs=1e-6)

    ve_inner_vol = ConvexHull(method.inner_vertices).volume
    cdd_inner_vol = ConvexHull(method_cdd.inner_vertices).volume
    assert ve_inner_vol == pytest.approx(cdd_inner_vol, rel=1e-4)
