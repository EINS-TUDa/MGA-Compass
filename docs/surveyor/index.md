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

- **[Explanation](explanation.md):** how the method works and why.
- **[How-to guides](how-to.md):** instructions for specific tasks.
- **[Reference](../reference/index.md):** the API.

```{toctree}
:hidden:

explanation
how-to
```
