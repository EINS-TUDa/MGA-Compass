"""``BilevelMethod`` -- "bilevel solution".

- Implements the method of Turan, Moret & Bardow, "ORACLE: A rigorous
  metric and method to explore all near-optimal designs for energy
  systems", Computers & Chemical Engineering (2026).
- Only ``norm='linf'`` is supported; L1/L2 aren't implemented (use
  :class:`~mga_surveyor.methods.cdd.CddMethod` instead).
- Requires ``gurobipy`` and a Gurobi license; no open-source backend.
- Reformulates the max-min bilevel LP into a single-level MILP via KKT +
  Big-M (Fortuny-Amat/McCarl) linearization of the complementarity
  conditions.
"""
import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    gp = None
    GRB = None

from ..common import Halfspace, Norm, check_norm
from .base import Method


def solve_farthest_outer_point_milp(
    A_O: np.ndarray, b_O: np.ndarray, inner_points: np.ndarray,
    M_P: float = 15.0, M_D: float = 30.0, verbose: bool = False,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Solve ``max_{zO in O} min_{zI in conv(inner_points)} ||zO - zI||_inf``
    via the KKT-based single-level MILP reformulation. ``O`` is given as
    ``A_O @ z <= b_O``.

    :param A_O: Outer polytope constraint matrix, with ``O`` given by
        ``A_O @ z <= b_O``.
    :param b_O: Outer polytope right-hand side, with ``O`` given by
        ``A_O @ z <= b_O``.
    :param inner_points: Points whose convex hull is the inner
        approximation.
    :param M_P: Big-M constant bounding the primal solution
        (Fortuny-Amat/McCarl linearization of the complementarity
        conditions); must be a valid (large enough) bound for the specific
        problem, or the reformulation is not exact -- there is no automatic
        way to check this from inside the MILP itself.
    :param M_D: Big-M constant bounding the dual solution; same caveat as
        ``M_P``.
    :param verbose: If True, let Gurobi print its own solve log.
    :returns: ``(z_O_star, t_star, w_star)`` -- the worst point, the max-min
        L_inf distance, and the convex-combination weights of its closest
        inner point (``z_I_star = w_star @ inner_points``).
    :rtype: tuple
    :raises ImportError: If ``gurobipy`` is not installed.
    """
    if gp is None:
        raise ImportError(
            "BilevelMethod requires the `gurobipy` package and a valid "
            "Gurobi license -- its MILP reformulation is Gurobi-specific, with no "
            "backend option (unlike CddMethod's own projection_backend; see module "
            "docstring). Install gurobipy, or use CddMethod instead "
            "(projection_backend='highs' needs no Gurobi at all, for norm in "
            "('l1', 'linf'))."
        )
    inner_points = np.asarray(inner_points, dtype=float)
    m, nz = inner_points.shape
    n_rows = A_O.shape[0]

    model = gp.Model()
    if not verbose:
        model.Params.OutputFlag = 0
    model.Params.MIPGap = 1e-7
    model.Params.IntFeasTol = 1e-9

    zO = model.addVars(nz, lb=-GRB.INFINITY, name="zO")
    t = model.addVar(lb=0.0, name="t")
    w = model.addVars(m, lb=0.0, name="w")

    a_plus = model.addVars(nz, lb=0.0, name="a_plus")
    a_minus = model.addVars(nz, lb=0.0, name="a_minus")
    b_dual = model.addVars(m, lb=0.0, name="b_dual")
    eta = model.addVar(lb=-GRB.INFINITY, name="eta")

    u_plus = model.addVars(nz, vtype=GRB.BINARY, name="u_plus")
    u_minus = model.addVars(nz, vtype=GRB.BINARY, name="u_minus")
    v_bin = model.addVars(m, vtype=GRB.BINARY, name="v")

    for r in range(n_rows):
        model.addConstr(gp.quicksum(A_O[r, j] * zO[j] for j in range(nz)) <= b_O[r])

    slack_plus, slack_minus = {}, {}
    for j in range(nz):
        model.addConstr(zO[j] - gp.quicksum(inner_points[i, j] * w[i] for i in range(m)) - t <= 0)
        slack_plus[j] = t - zO[j] + gp.quicksum(inner_points[i, j] * w[i] for i in range(m))

        model.addConstr(-zO[j] + gp.quicksum(inner_points[i, j] * w[i] for i in range(m)) - t <= 0)
        slack_minus[j] = t + zO[j] - gp.quicksum(inner_points[i, j] * w[i] for i in range(m))

    model.addConstr(gp.quicksum(w[i] for i in range(m)) == 1, name="sum_to_one")

    model.addConstr(gp.quicksum(a_plus[j] + a_minus[j] for j in range(nz)) == 1, name="stat_t")
    for i in range(m):
        model.addConstr(
            eta - b_dual[i] - gp.quicksum(inner_points[i, j] * (a_plus[j] - a_minus[j]) for j in range(nz)) == 0,
            name=f"stat_w_{i}",
        )

    for j in range(nz):
        model.addConstr(a_plus[j] <= u_plus[j] * M_D)
        model.addConstr(slack_plus[j] <= (1 - u_plus[j]) * M_P)
        model.addConstr(a_minus[j] <= u_minus[j] * M_D)
        model.addConstr(slack_minus[j] <= (1 - u_minus[j]) * M_P)
    for i in range(m):
        model.addConstr(b_dual[i] <= v_bin[i] * M_D)
        model.addConstr(w[i] <= (1 - v_bin[i]) * M_P)

    # tightening from Caratheodory + L_inf structure (optional but cheap)
    model.addConstr(gp.quicksum(1 - v_bin[i] for i in range(m)) <= nz + 1)
    model.addConstr(gp.quicksum(1 - v_bin[i] for i in range(m)) >= 1)
    model.addConstr(gp.quicksum(u_plus[j] + u_minus[j] for j in range(nz)) >= 1)
    model.addConstr(gp.quicksum(u_plus[j] + u_minus[j] for j in range(nz)) <= nz)

    model.setObjective(t, GRB.MAXIMIZE)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"MILP failed: status {model.Status}")

    z_O_star = np.array([zO[j].X for j in range(nz)])
    t_star = t.X
    w_star = np.array([w[i].X for i in range(m)])
    return z_O_star, t_star, w_star


class BilevelMethod(Method):
    """See module docstring.

    :param k: Dimension of the exploratory variables.
    :param norm: Must be ``'linf'`` -- see module docstring for why.
    :type norm: str
    :param M_P: Big-M primal bound, forwarded to
        :func:`solve_farthest_outer_point_milp` on every call.
    :param M_D: Big-M dual bound, forwarded to
        :func:`solve_farthest_outer_point_milp` on every call.
    :param verbose: If True, let Gurobi print its own solve log.
    :raises NotImplementedError: If ``norm`` is not ``'linf'``.
    """

    def __init__(self, k: int, norm: Norm, M_P: float = 15.0, M_D: float = 30.0, verbose: bool = False) -> None:
        check_norm(norm)
        if norm != "linf":
            raise NotImplementedError(
                "BilevelMethod only supports norm='linf' -- see module docstring "
                "for why L1/L2 would each need a different reformulation. "
                "Use CddMethod for other norms."
            )
        super().__init__(k, norm)
        self.M_P = M_P
        self.M_D = M_D
        self.verbose = verbose

        self.inner_vertices = None
        self.outer_halfspaces = []
        self._dirty = True
        self._trial_point = None
        self._gap = 0.0

    def bootstrap(self, initial_points: list[np.ndarray], initial_halfspaces: list[Halfspace]) -> None:
        self.inner_vertices = np.array(initial_points, dtype=float)
        self.outer_halfspaces = list(initial_halfspaces)
        self._dirty = True

    def add_point(self, point: np.ndarray) -> bool:
        point = np.asarray(point, dtype=float)
        is_new = np.linalg.norm(self.inner_vertices - point, axis=1).min() > 1e-6
        if is_new:
            self.inner_vertices = np.vstack([self.inner_vertices, point])
            self._dirty = True
        return is_new

    def add_cut(self, direction: np.ndarray, point: np.ndarray) -> None:
        direction = np.asarray(direction, dtype=float)
        point = np.asarray(point, dtype=float)
        self.outer_halfspaces = self.outer_halfspaces + [(direction, point)]
        self._dirty = True

    def _matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Assemble ``(A_O, b_O)`` from ``self.outer_halfspaces`` for
        :func:`solve_farthest_outer_point_milp`.

        :returns: ``(A_O, b_O)`` such that ``O`` is ``A_O @ z <= b_O``.
        :rtype: tuple
        """
        A_O = np.array([direction for direction, _ in self.outer_halfspaces])
        b_O = np.array([direction @ point for direction, point in self.outer_halfspaces])
        return A_O, b_O

    def find_trial_point(self) -> tuple[np.ndarray | None, float]:
        """Cached: recomputes only if ``self._dirty`` is set (by
        :meth:`bootstrap`, :meth:`add_point`, or :meth:`add_cut` since the
        last call); a repeat call in between -- notably from
        :meth:`max_gap` -- skips the MILP solve and reuses the cached
        result.
        """
        if not self._dirty:
            return self._trial_point, self._gap
        if not self.outer_halfspaces:
            self._trial_point, self._gap = None, 0.0
        else:
            A_O, b_O = self._matrices()
            z_O_star, t_star, _w_star = solve_farthest_outer_point_milp(
                A_O, b_O, self.inner_vertices, M_P=self.M_P, M_D=self.M_D, verbose=self.verbose,
            )
            self._trial_point, self._gap = z_O_star, float(t_star)
        self._dirty = False
        return self._trial_point, self._gap

    def max_gap(self) -> float:
        _point, gap = self.find_trial_point()
        return gap
