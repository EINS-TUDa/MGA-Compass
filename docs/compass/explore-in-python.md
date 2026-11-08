# Explore a near-optimal set in Python

The web platform is one way to explore a near-optimal set; the `mga_compass` library is
another, for notebooks and scripts. This guide loads a set, navigates from the cheapest design
to one with more wind, and looks at the cost and the system along the way. The same steps,
with plots, are in the notebook
[`mga-compass/examples/explore-a-near-optimal-set.ipynb`](https://github.com/EINS-TUDa/MGA-Compass/blob/main/mga-compass/examples/explore-a-near-optimal-set.ipynb).

You need:

- the `mga-compass` library, with the `notebook` extra for plots:
  `uv sync --package mga-compass --extra notebook` (see [](install.md));
- a [near-optimal set](../reference/near-optimal-set-format.md): a directory, its zip
  archive, or the URL of a published one. This guide uses the published set of PyPSA's
  `model_energy` example, which `scripts/generate_dataset.py` builds (see [](build-dataset.md)).

## Load the set

```python
import mga_compass as mc

URL = "https://github.com/EINS-TUDa/MGA-Compass/releases/download/dataset-model-energy-2026-11-07/near-optimal-set.zip"
space = mc.load(URL, "data/model-energy")
space
```

```text
PyPSA model_energy example
PyPSA's model_energy example (pypsa.examples.model_energy): one electricity node and one
hydrogen node over 2019, at 9-hourly resolution (every third snapshot of the 3-hourly example).
...

Space: 188 near-optimal designs in 4 PoIs, slack 0.08 (set's slack levels: 0.02, 0.05, 0.08)
         unit     min    max optimum  description
wind       GW   15.35  42.04   25.27  Installed wind power capacity.
solar      GW   13.08  55.38   30.53  Installed solar PV capacity.
battery    GW   6.632  35.47   16.83  Installed battery storage power (charge/discharge) rating. ...
hydrogen  TWh  0.4023  5.349   1.501  Installed hydrogen storage capacity.
TOTEX      M€   6,901  7,453   6,901
```

A `Space` holds the set's designs in the space of its properties of interest (PoIs), with
their cost. `space.points` has one row per design; `space.optimum` is the cheapest.

- **Units:** values are in the set's display units (here capacities in GW and TWh rather
  than MW and MWh, the cost in M€ rather than €). With `units="model"`, they stay in the
  model's units. Navigation finds the same designs either
  way.
- **Slack level:** by default the space is the set's largest slack level. `slack=0.02`
  keeps only the designs found within 2% of the optimal cost.
- **Archives:** a zip archive or a URL is unpacked into the directory you name, from which
  outputs are read later. Once unpacked, load it from there: `mc.load("data/model-energy")`.
- **About the set:** `space.title` and `space.description` say what model it comes from,
  when the set's manifest gives them.

## Navigate

Ask for at least 20% more wind capacity, with the cost at most 3% above the start's:

```python
result = space.navigate(space.optimum, {
    "wind": ("increase", "20%"),
    "TOTEX": ("keep", "3%"),
})
result
```

```text
Navigation: 2 requests, 0 adjusted
         unit  start  target    change                request
wind       GW  25.27   34.82    +9.544   increase by 5.054 GW
solar      GW  30.53   20.39    -10.15
battery    GW  16.83   16.76  -0.06618
hydrogen  TWh  1.501  0.8796   -0.6211
TOTEX      M€  6,901   7,108    +207.0  keep within ±207.0 M€
Traverse: 2 breakpoints between start and target; the cost is at most 34.27 M€ above its lower bound
```

A request maps a PoI, or the cost, to a direction and a delta:

| Request | Means |
|---|---|
| `("increase", d)` | At least `d` more, then as much more as the requests before it allow. |
| `("decrease", d)` | At least `d` less, then as much less as the requests before it allow. |
| `("keep", d)` | Within `d` of the start value, then as close to it as the requests before it allow. |
| `"keep"`, `"increase"`, `"decrease"` | The same with `d = 0`. |

`d` is in the space's unit, or a percentage of the start value such as `"20%"`. The order of
the requests is their priority. PoIs not listed are free, and move however is cheapest: here
wind replaces solar and hydrogen storage. Wind grew by more than asked, as far as the cost
limit allowed.

When a request cannot be met in full, it is met as far as possible, and `result.changes` says
how, for example `wind: could increase by 6.148 GW only, not 7.596 GW`.

To carry on, navigate from the result: `space.navigate(result, {...})`. A point id, or any
row of `space.points`, works as a start too.

## Look at the way there

`result.traverse` is the way from the start to the target. The PoIs move in a straight line;
the cost is the cheapest mix of the set's designs at each point, an upper bound on the true
cost, and the designs' duals give a lower bound (see [](traverse.md)):

```python
result.traverse
```

```text
Traverse: 2 breakpoints between start and target; the cost is at most 34.27 M€ above its lower bound
       wind  solar battery hydrogen  TOTEX lower bound
beta
0     25.27  30.53   16.83    1.501  6,901       6,901
0.5   30.04  25.46   16.80    1.190  6,991       6,963
0.75  32.43  22.92   16.78    1.035  7,036       7,007
1     34.82  20.39   16.76   0.8796  7,108       7,099
```

- `result.traverse.at(0.3)`: the design at 30% of the way, with its cost.
- `result.traverse.lower_bound_at(0.3)`: the cost's lower bound there.
- `result.traverse.plot()`: both bounds against β.
- `space.traverse(a, b)`: the way between any two designs.

## Look at the system

The set's detailed outputs are listed by `space.outputs()`. `space.output` gives one of them
for a design: a navigation result (its target), the weights of a point along the traverse,
or a design of the set by its id. It reads only the outputs of the designs in the mix.

```python
import pandas as pd

pd.DataFrame({
    "start": space.output("capacity", result.traverse.alpha_at(0)),
    "halfway": space.output("capacity", result.traverse.alpha_at(0.5)),
    "target": space.output("capacity", result),
})
```

```text
                       start     halfway    target
technology
wind                25,271.6    30,043.7  34,815.9
solar               30,534.5    25,460.4  20,386.3
electrolysis         1,709.7     1,450.7   1,312.7
turbine              9,904.6     9,584.5   9,296.5
battery storage     16,828.8    16,795.7  16,762.6
hydrogen storage 1,500,715.2 1,190,180.7 879,645.2
```

An output is a pandas Series indexed by the output's sets, with its unit in
`series.attrs["unit"]`. Unstack it for a table, and plot it with pandas, e.g. the target's
generation as weekly means:

```python
space.output("generation", result).unstack().resample("W").mean().plot.area()
```

## See also

- {py:mod}`mga_compass.space` in the [API reference](../reference/compass-api.md).
- [](navigation.md) and [](traverse.md): how navigation and the traverse work.
