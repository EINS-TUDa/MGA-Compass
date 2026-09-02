import numpy as np

from mga_surveyor.methods.vertex_enum import VertexEnumMethod


def _stub_projection_fn(point, points, norm):
    """Not exercised by these tests -- add_cut's redundancy detection acts
    purely on whether the new halfspace removes any current outer vertex,
    independent of the inner hull's own closest points/normals."""
    del points, norm
    return np.zeros_like(point), np.zeros_like(point)


def _unit_square_method() -> VertexEnumMethod:
    method = VertexEnumMethod(k=2, norm="linf", projection_fn=_stub_projection_fn)
    box_halfspaces = [
        (np.array([1.0, 0.0]), np.array([1.0, 0.0])),   # x <= 1
        (np.array([-1.0, 0.0]), np.array([-1.0, 0.0])),  # x >= -1
        (np.array([0.0, 1.0]), np.array([0.0, 1.0])),   # y <= 1
        (np.array([0.0, -1.0]), np.array([0.0, -1.0])),  # y >= -1
    ]
    method.bootstrap([np.array([0.0, 0.0])], box_halfspaces)
    return method


def test_add_cut_discards_a_redundant_halfspace():
    """x <= 2 is already implied by the square's own x <= 1 -- removes no
    current vertex, so it must be detected as redundant and NOT appended
    to outer_halfspaces, leaving the vertex set untouched."""
    method = _unit_square_method()
    n_halfspaces_before = len(method.outer_halfspaces)
    n_vertices_before = len(method.outer_vertices)

    result = method.add_cut(np.array([1.0, 0.0]), np.array([2.0, 0.0]))

    assert result == (0, 0)
    assert len(method.outer_halfspaces) == n_halfspaces_before
    assert len(method.outer_vertices) == n_vertices_before


def test_add_cut_keeps_a_genuine_halfspace():
    """x <= 0 actually slices the square in half -- must remove vertices,
    get appended to outer_halfspaces, and leave every surviving vertex
    satisfying the new constraint."""
    method = _unit_square_method()
    n_halfspaces_before = len(method.outer_halfspaces)

    n_new, n_removed = method.add_cut(np.array([1.0, 0.0]), np.array([0.0, 0.0]))

    assert n_removed > 0
    assert n_new > 0
    assert len(method.outer_halfspaces) == n_halfspaces_before + 1
    assert np.all(method.outer_vertices[:, 0] <= 1e-6)


def test_add_cut_redundancy_is_exact_not_approximate():
    """A halfspace exactly touching the square's own boundary at a single
    existing vertex (e.g. x + y <= 2, tight only at (1, 1)) removes no
    vertex (every vertex satisfies x + y <= 2, with equality at one) --
    still redundant, still discarded."""
    method = _unit_square_method()
    n_halfspaces_before = len(method.outer_halfspaces)

    result = method.add_cut(np.array([1.0, 1.0]), np.array([1.0, 1.0]))  # x + y <= 2

    assert result == (0, 0)
    assert len(method.outer_halfspaces) == n_halfspaces_before
