# MGA Compass

Near-optimal solutions cost a small percentage more than the optimum. While slightly
expensive, they can be preferable for political feasibility, social acceptance, or
environmental impact.

This project contains two parts:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} ![](mga_compass_logo.svg){height=28px} MGA Compass
:link: compass/index
:link-type: doc

An interactive platform to explore the near-optimal solutions of an energy system model
in real time.
:::

:::{grid-item-card} ![](mga_surveyor_logo.svg){height=28px} MGA Surveyor
:link: surveyor/index
:link-type: doc

The implementation of the G-ORACLE method for [PyPSA](https://pypsa.org): computes the
near-optimal space of a PyPSA model, with a guaranteed bound on how much of it is still
unexplored.
:::
::::

MGA Surveyor finds the near-optimal solutions; MGA Compass lets you explore them. The two
libraries are independent, and the platform can serve points from any source.

**Live demo:**
[enormous-raphaela-eins-tu-darmstadt-215a3282.koyeb.app](https://enormous-raphaela-eins-tu-darmstadt-215a3282.koyeb.app/)

## Presentations

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Trondheim
:link: presentations/Trondheim/
:link-type: url

EEM 2026
+++
View slides →
:::

:::{grid-item-card} Freiburg
:link: presentations/Freiburg/
:link-type: url

openmod Workshop 2026
+++
View slides →
:::
::::

## Citation

If you use **MGA Compass**, please cite:

S. Hajikazemi and F. Steinke, "Interactive Exploration of Near-Optimal Solutions of Energy
Planning Models," in *2026 22nd International Conference on the European Energy Market
(EEM)*, 2026, pp. 1–6.
[doi:10.1109/EEM68581.2026.11589682](https://doi.org/10.1109/EEM68581.2026.11589682)

```bibtex
@inproceedings{hajikazemi2026compass,
  title     = {Interactive Exploration of Near-Optimal Solutions of Energy Planning Models},
  author    = {Hajikazemi, Sina and Steinke, Florian},
  booktitle = {2026 22nd International Conference on the European Energy Market (EEM)},
  year      = {2026},
  pages     = {1--6},
  doi       = {10.1109/EEM68581.2026.11589682}
}
```

If you use **MGA Surveyor** or G-ORACLE, please cite:

S. Hajikazemi and F. Steinke, "G-ORACLE: A Fast Geometric Method To Explore All
Near-Optimal Designs for Energy Systems," arXiv:2610.32759, 2026.
[arxiv.org/abs/2610.32759](https://arxiv.org/abs/2610.32759)

```bibtex
@misc{hajikazemi2026goracle,
  title         = {G-ORACLE: A Fast Geometric Method To Explore All Near-Optimal Designs for Energy Systems},
  author        = {Hajikazemi, Sina and Steinke, Florian},
  year          = {2026},
  eprint        = {2610.32759},
  archivePrefix = {arXiv},
  primaryClass  = {math.OC},
  url           = {https://arxiv.org/abs/2610.32759}
}
```

```{toctree}
:hidden:
:maxdepth: 2

surveyor/index
compass/index
reference/index
```
