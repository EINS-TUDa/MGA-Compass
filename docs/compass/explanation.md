# Explanation

How MGA Compass works and why. The approach has two phases. A **preparation phase** solves
the full model a number of times to collect near-optimal sample points; it is expensive but
runs once per model. The **exploration phase** then works only with those samples, so every
interaction solves small problems whose size does not depend on the model, and answers within
seconds.

These pages follow the MGA Compass paper (Hajikazemi and Steinke, EEM 2026,
[doi:10.1109/EEM68581.2026.11589682](https://doi.org/10.1109/EEM68581.2026.11589682)). They
use the same notation as the [MGA Surveyor pages](../surveyor/near-optimal-set.md); where the
paper writes $p$, $\mathcal{P}^{\mathrm{R}}$ and $\phi$, these pages write $z$, $\mathcal{Z}$
and $\nu$.

```{toctree}
:maxdepth: 1

preparation
navigation
traverse
```
