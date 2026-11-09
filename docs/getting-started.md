# Getting started

In this tutorial you explore the near-optimal designs of a small energy system: you compute
them with MGA Surveyor, then move from the cheapest design to one with more wind power with
MGA Compass, and see what that costs.

It takes about 10 minutes, including about 2 minutes of computing. Everything runs on the
open-source solver HiGHS, so you need no licence.

## What you need

- [uv](https://docs.astral.sh/uv/) and git.
- An internet connection: the example model is downloaded by PyPSA.

Clone the repository and install both packages with PyPSA:

```bash
git clone https://github.com/EINS-TUDa/MGA-Compass.git
cd MGA-Compass
uv sync --package mga-compass --extra scripts
```

Create a file `tutorial.py` in the repository root. You will add to it step by step and run
it at the end.

## Step 1: Solve the model

The model is PyPSA's `model_energy` example: a single-node system with wind, solar, battery
and hydrogen storage. To keep the tutorial fast, it uses one time step per day.

```python
from pathlib import Path

import pandas as pd
import pypsa

from mga_compass import Constraints, default_solver_options, interpolate, navigate
from mga_compass.near_optimal import navigable_points
from mga_surveyor import collect_near_optimal_set, generate_mga_files

pd.set_option("display.float_format", "{:,.1f}".format)  # plain numbers, not 3.9e+09


def main() -> None:
    n = pypsa.examples.model_energy()
    n.snapshots = n.snapshots[::24]  # one time step per day
    n.optimize(solver_name="highs", include_objective_constant=True)
    print(f"optimal cost: {n.objective:,.0f}")
```

## Step 2: Find the near-optimal designs

Choose what you want to compare designs by, here the installed capacity of each technology,
and let MGA Surveyor explore every design that costs at most 5% more than the optimum. Add
this to `main()`:

```python
    dimensions = {
        "wind": {"Generator": {"p_nom": {"wind": 1}}},
        "solar": {"Generator": {"p_nom": {"solar": 1}}},
        "battery": {"StorageUnit": {"p_nom": {"battery storage": 1}}},
        "hydrogen": {"Store": {"e_nom": {"hydrogen storage": 1}}},
    }
    generate_mga_files(
        n,
        dimensions,
        Path("tutorial-output"),
        norm="linf",
        slack_levels=[0.05],
        solver_name="highs",
        tol=5,
        max_iter=50,
        max_parallel=2,
    )
```

`tol=5` stops the exploration once every capacity is known to within 5% of its range.

## Step 3: Collect the designs

Read the results back as a near-optimal set, giving the unit of each capacity and of the
cost, and take its designs as a table: one row per design, one column per capacity, plus the
cost.

```python
    nos, _ = collect_near_optimal_set(
        "tutorial-output",
        units={"wind": "MW", "solar": "MW", "battery": "MW", "hydrogen": "MWh"},
        cost_unit="EUR",
        cost_name="TOTEX",
    )
    points = navigable_points(nos)
    print(f"{len(points)} designs")
    print(points.round(1).head())
```

The values stay in the model's units, euros next to megawatts: MGA Compass handles columns
of very different sizes itself.

## Step 4: Ask for more wind

Start from the cheapest design and ask for at least 20% more wind capacity, while letting the
cost rise by at most 3%. The order of the rows is the priority order: wind first, then cost.
The other capacities are left free (`""`).

```python
    start = points.loc[points["TOTEX"].idxmin()]
    preferences = Constraints(pd.DataFrame({
        "direction": [">=", "==", "", "", ""],
        "value": [start["wind"], start["TOTEX"], start["solar"], start["battery"], start["hydrogen"]],
        "delta": [0.2 * start["wind"], 0.03 * start["TOTEX"], 0, 0, 0],
    }, index=["wind", "TOTEX", "solar", "battery", "hydrogen"]))

    solver_options = default_solver_options("highs")
    adjusted, target = navigate(points, preferences, obj_label="TOTEX", solver_options=solver_options)
    print(pd.DataFrame({"start": start, "target": target}).round(1))
```

## Step 5: Look at the way there

Compute the path from the cheapest design to the new one, with the cost along the way:

```python
    breakpoints = interpolate(points, start, target, obj_label="TOTEX", solver_options=solver_options)
    for bp in breakpoints:
        print(f"at {bp.beta:.0%} of the way: {bp.point['TOTEX'] / 1e6:,.1f} million EUR")


if __name__ == "__main__":
    main()
```

The `if __name__ == "__main__":` guard is needed because `generate_mga_files` starts worker
processes.

## Run it

```bash
uv run python tutorial.py
```

After about 2 minutes, the end of the output looks like this:

```text
38 designs
          wind   solar  battery  hydrogen           TOTEX
point
0     18,092.5     0.0    288.9       0.0 3,899,079,889.9
1     18,092.5 3,796.8    288.9       0.0 4,094,032,909.6
...
                   start          target
wind            18,092.5        24,654.0
solar                0.0             0.0
battery            288.9         2,013.4
hydrogen             0.0             0.0
TOTEX    3,899,079,889.9 4,016,052,286.6
at 0% of the way: 3,899.1 million EUR
at 100% of the way: 4,016.1 million EUR
```

The number of designs, and which ones appear first, can differ from run to run, because the
model solves run in parallel and finish in varying order. The start and target designs are
the same every time.

## What you have seen

- MGA Surveyor found about 40 near-optimal designs, all within 5% of the optimal cost of
  3,899 million EUR.
- Asked for at least 20% more wind for at most 3% more cost, MGA Compass found a design with
  36% more wind (18.1 → 24.7 GW). It also adds battery power (0.3 → 2.0 GW) to absorb the
  extra wind, and costs exactly 3% more.
- The path between the two designs needed no breakpoints in between: going straight from one
  to the other is already the cheapest way.

## Next steps

- Understand what happened: [the near-optimal set](surveyor/near-optimal-set.md),
  [G-ORACLE](surveyor/g-oracle.md), [navigation](compass/navigation.md) and
  [traverse](compass/traverse.md).
- Run it on your own model: [](surveyor/explore-network.md).
- Explore your results in the web platform: [](compass/build-dataset.md).
