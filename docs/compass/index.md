# MGA Compass

MGA Compass is an interactive platform for exploring the near-optimal solutions of an energy
system model in real time. You start from a solution, say how the next one should differ in
a few properties you care about, and the platform finds it and shows the transition along an
approximately cost-optimal path.

It consists of a web application (`app/`: backend and frontend) and the `mga-compass` Python
library, which holds the navigation, interpolation and plotting logic the platform runs on.
The library has no GUI or server dependencies.

**Live demo:**
[enormous-raphaela-eins-tu-darmstadt-215a3282.koyeb.app](https://enormous-raphaela-eins-tu-darmstadt-215a3282.koyeb.app/)

The demo shows the solutions of [PyPSA-Eur](https://github.com/PyPSA/pypsa-eur) within 5% of
the optimal cost: electricity only, one node per country (34 nodes), one year at 3-hourly
resolution.

![MGA Compass](../MGA-compass.png)

## Installation

From a clone of the repository, using [uv](https://docs.astral.sh/uv/):

```bash
uv sync --package mga-compass
```

## How it works

The approach has two phases. A **preparation phase** solves the full model a number of times
to collect near-optimal sample points; it is expensive but runs once per model. The
**exploration phase** then works only with those samples, so every interaction solves small
problems whose size does not depend on the model, and answers within seconds.

```{toctree}
:maxdepth: 1

preparation
navigation
traverse
```

These pages follow the MGA Compass paper (Hajikazemi and Steinke, EEM 2026,
[doi:10.1109/EEM68581.2026.11589682](https://doi.org/10.1109/EEM68581.2026.11589682)). They
use the same notation as the [MGA Surveyor pages](../surveyor/near-optimal-set.md); where the
paper writes $p$, $\mathcal{P}^{\mathrm{R}}$ and $\phi$, these pages write $z$, $\mathcal{Z}$
and $\nu$.

For the full interface, see the [API reference](../reference/compass-api.md).
