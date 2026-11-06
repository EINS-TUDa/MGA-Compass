# MGA Surveyor

MGA Surveyor computes the **near-optimal set** of a [PyPSA](https://pypsa.org) model: every
design whose cost lies within a given slack of the optimum, described by a few properties of
interest such as the installed capacity of each technology. It implements **G-ORACLE**, an
MGA method that refines an inner and an outer polytope approximation of that set until the
two are within a chosen tolerance of each other, so you always know how much of the set is
still unexplored.

```{image} ../g-oracle.svg
:alt: One G-ORACLE iteration, animated
:width: 300px
:align: center
```

## Installation

From a clone of the repository, using [uv](https://docs.astral.sh/uv/):

```bash
uv sync --package mga-surveyor
```

`gurobipy` and `highspy` are installed by default. `CddMethod` additionally needs the `cdd`
extra, built from the patched, vendored `pycddlib`; see `mga-surveyor/vendor/cdd/README.md`.

## How it works

These pages explain the method behind the package and how each part maps onto the code.
They follow the G-ORACLE paper
([arXiv:2610.32759](https://arxiv.org/abs/2610.32759)), which has the full proofs and the
numerical experiments.

```{toctree}
:maxdepth: 1

near-optimal-set
oracle
g-oracle
methods
normalization
```

For the full interface, see the [API reference](../reference/surveyor-api.md).
