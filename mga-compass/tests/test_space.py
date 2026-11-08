import numpy as np
import pandas as pd
import pytest

import mga_compass as mc
import mga_format

LEVEL = {"norm": "linf", "normalize": True, "tol": 5, "tol_unit": "percent",
         "vertex_method": "bb", "objective_cuts": True, "iterations": 1, "gap": 1, "converged": True}


@pytest.fixture
def set_dir(tmp_path):
    """Two PoIs, two slack levels: the optimum, two designs at 1% and two at 5%."""
    manifest = mga_format.Manifest.model_validate({
        "format": "mga-near-optimal-set", "version": 1, "created": "2026-11-07T12:00:00+00:00",
        "cost": {"name": "TOTEX", "unit": "EUR", "optimum": 1000.0},
        "pois": [{"name": "wind", "unit": "MW", "description": "wind capacity", "definition": {}},
                 {"name": "solar", "unit": "MW", "definition": {}}],
        "slack_levels": [{"slack": 0.01, **LEVEL}, {"slack": 0.05, **LEVEL}],
        "fingerprint": {"network": "x"},
        "display": {"wind": {"unit": "GW", "factor": 1e-3}, "TOTEX": {"unit": "kEUR", "factor": 1e-3},
                    "capacity": {"unit": "GW", "factor": 1e-3}},
        "sets": {"technology": {}},
        "outputs": {"capacity": {"index": ["point", "technology"], "unit": "MW", "description": "capacity per technology"}},
    })
    index = pd.Index([0, 1, 2, 3, 4], name="point")
    points = pd.DataFrame({
        "slack": [0.0, 0.01, 0.01, 0.05, 0.05],
        "kind": ["optimum", "box", "box", "box", "nearest"],
        "wind": [2000.0, 2500.0, 1500.0, 4000.0, 1000.0],
        "solar": [500.0, 400.0, 700.0, 100.0, 900.0],
        "TOTEX": [1000.0, 1010.0, 1008.0, 1050.0, 1040.0],
    }, index=index)
    duals = pd.DataFrame({"wind": [0.0] * 5, "solar": [0.0] * 5}, index=index)
    # The output repeats the PoIs: each design's wind and solar capacity.
    capacity = points[["wind", "solar"]].rename_axis(columns="technology").stack().rename("value").reset_index()
    sets = {"technology": pd.DataFrame({"technology": ["wind", "solar"]})}
    directory = tmp_path / "set"
    mga_format.write(directory, mga_format.NearOptimalSet(manifest, points, duals, sets), {"capacity": capacity})
    return directory


def test_load_directory_in_display_units(set_dir):
    space = mc.load(set_dir)
    assert space.pois == ["wind", "solar"]
    assert space.cost == "TOTEX"
    assert space.units == {"wind": "GW", "solar": "MW", "TOTEX": "kEUR"}
    assert list(space.points.columns) == ["wind", "solar", "TOTEX"]
    assert list(space.points["wind"]) == pytest.approx([2.0, 2.5, 1.5, 4.0, 1.0])
    assert list(space.points["solar"]) == [500.0, 400.0, 700.0, 100.0, 900.0]  # no display entry
    assert space.slack == 0.05
    assert space.directory == set_dir


def test_model_units(set_dir):
    space = mc.load(set_dir, units="model")
    assert space.units == {"wind": "MW", "solar": "MW", "TOTEX": "EUR"}
    assert list(space.points["wind"]) == [2000.0, 2500.0, 1500.0, 4000.0, 1000.0]


def test_unknown_units(set_dir):
    with pytest.raises(ValueError, match="units"):
        mc.load(set_dir, units="SI")


def test_smaller_slack_keeps_optimum_and_designs_up_to_it(set_dir):
    space = mc.load(set_dir, slack=0.01)
    assert space.slack == 0.01
    assert list(space.points.index) == [0, 1, 2]


def test_slack_must_be_a_level_of_the_set(set_dir):
    with pytest.raises(ValueError, match="slack levels"):
        mc.load(set_dir, slack=0.03)


def test_optimum_is_the_cheapest_design(set_dir):
    optimum = mc.load(set_dir).optimum
    assert optimum.name == 0
    assert optimum["TOTEX"] == pytest.approx(1.0)


def test_summary(set_dir):
    summary = mc.load(set_dir).summary()
    assert list(summary.index) == ["wind", "solar", "TOTEX"]
    assert summary.loc["wind", "unit"] == "GW"
    assert summary.loc["wind", ["min", "max", "optimum"]].tolist() == pytest.approx([1.0, 4.0, 2.0])
    assert summary.loc["wind", "description"] == "wind capacity"
    assert summary.loc["TOTEX", "description"] == ""


def test_repr_shows_designs_slack_and_units(set_dir):
    text = repr(mc.load(set_dir, units="model"))
    assert "5 near-optimal designs in 2 PoIs, slack 0.05" in text
    assert "0.01, 0.05" in text
    assert "4,000" in text and "MW" in text and "wind capacity" in text
    assert "<table" in mc.load(set_dir)._repr_html_()


def test_title_and_description_from_the_manifest(set_dir, tmp_path):
    space = mc.load(set_dir)
    assert (space.title, space.description) == ("", "")
    assert repr(space).startswith("Space: ")
    nos = mga_format.read(set_dir)
    manifest = nos.manifest.model_copy(update={"title": "Toy model", "description": "Two PoIs & one node."})
    outputs = {"capacity": mga_format.read_output(set_dir, "capacity")}
    mga_format.write(tmp_path / "described", mga_format.NearOptimalSet(manifest, nos.points, nos.duals, nos.sets), outputs)
    space = mc.load(tmp_path / "described")
    assert (space.title, space.description) == ("Toy model", "Two PoIs & one node.")
    assert repr(space).startswith("Toy model\nTwo PoIs & one node.\n\nSpace: ")
    assert "<h4>Toy model</h4><p>Two PoIs &amp; one node.</p>" in space._repr_html_()


def test_load_archive_into_directory(set_dir, tmp_path):
    archive = mga_format.pack(set_dir, tmp_path / "set.zip")
    space = mc.load(archive, tmp_path / "unpacked")
    assert space.directory == tmp_path / "unpacked"
    assert len(space.points) == 5


def test_load_url(set_dir, tmp_path):
    archive = mga_format.pack(set_dir, tmp_path / "set.zip")
    space = mc.load(archive.as_uri(), tmp_path / "fetched")
    assert len(space.points) == 5


def test_archive_needs_directory(set_dir, tmp_path):
    archive = mga_format.pack(set_dir, tmp_path / "set.zip")
    with pytest.raises(ValueError, match="directory"):
        mc.load(archive)


def test_directory_takes_no_second_directory(set_dir, tmp_path):
    with pytest.raises(ValueError, match="only for archives"):
        mc.load(set_dir, tmp_path / "elsewhere")


# ---------------------------------------------------------------- navigate
#
# In model units, the most wind for a given cost runs from the optimum (2000 MW,
# 1000 EUR) to point 1 (2500, 1010), then to point 3 (4000, 1050): 37.5 MW per
# EUR on that second stretch, where solar falls from 400 to 100 MW.


@pytest.fixture
def space(set_dir):
    return mc.load(set_dir, units="model")


def test_navigate_meets_requests_at_least_cost(space):
    result = space.navigate(space.optimum, {"wind": ("increase", 1000), "TOTEX": ("keep", 30)})
    # wind >= 3000 needs at most 1023.3 EUR; then as much wind as cost <= 1030 allows.
    assert result.target["wind"] == pytest.approx(3250)
    assert result.target["TOTEX"] == pytest.approx(1030)
    assert result.target["solar"] == pytest.approx(250)
    assert result.changes == []
    assert result.alpha.sum() == pytest.approx(1)
    assert list(result.target.index) == ["wind", "solar", "TOTEX"]


def test_request_order_is_priority(space):
    wind_first = space.navigate(space.optimum, {"wind": ("increase", 3000), "TOTEX": ("keep", 30)})
    assert wind_first.target["wind"] == pytest.approx(4000)
    assert wind_first.target["TOTEX"] == pytest.approx(1050)
    assert wind_first.changes == [
        "wind: could increase by 2,000 MW only, not 3,000 MW",
        "TOTEX: could not be kept within ±30.00 EUR; kept within ±50.00 EUR",
    ]
    cost_first = space.navigate(space.optimum, {"TOTEX": ("keep", 30), "wind": ("increase", 3000)})
    assert cost_first.target["wind"] == pytest.approx(3250)
    assert cost_first.changes == ["wind: could increase by 1,250 MW only, not 3,000 MW"]


def test_could_not_move_at_all(space):
    # Cost kept at the optimum's: no design but the optimum is that cheap.
    result = space.navigate(space.optimum, {"TOTEX": "keep", "wind": ("decrease", 100)})
    assert result.target["wind"] == pytest.approx(2000)
    assert result.changes == ["wind: could not decrease at all, given the requests before it"]


def test_percent_delta(space):
    result = space.navigate(space.optimum, {"wind": ("increase", "50%"), "TOTEX": ("keep", "3%")})
    assert result.requested.loc["wind", "delta"] == pytest.approx(1000)
    assert result.requested.loc["TOTEX", "delta"] == pytest.approx(30)
    assert result.target["wind"] == pytest.approx(3250)


def test_unlisted_pois_are_free_and_follow_the_requests(space):
    result = space.navigate(space.optimum, {"wind": ("increase", 1000)})
    assert list(result.requested.index) == ["wind", "solar", "TOTEX"]
    assert result.requested.loc["solar", "direction"] == ""
    assert result.target["wind"] == pytest.approx(4000)  # pushed as far as possible


def test_carry_on_from_a_result_or_a_point_id(space):
    first = space.navigate(space.optimum, {"wind": ("increase", 1000), "TOTEX": ("keep", 30)})
    second = space.navigate(first, {"solar": ("increase", 100), "wind": "keep"})
    assert second.start.equals(first.target.rename("start"))
    assert second.target["solar"] == pytest.approx(350)
    from_id = space.navigate(1, {"wind": "keep", "TOTEX": "decrease"})
    assert from_id.start["wind"] == 2500
    assert from_id.target["TOTEX"] == pytest.approx(1010)


def test_same_designs_in_display_units(set_dir):
    display = mc.load(set_dir)
    result = display.navigate(display.optimum, {"wind": ("increase", 1), "TOTEX": ("keep", 0.03)})
    assert result.target["wind"] == pytest.approx(3.25)
    assert result.target["TOTEX"] == pytest.approx(1.03)


@pytest.mark.parametrize("request_, match", [
    ({"nuclear": ("increase", 1)}, "neither a PoI nor the cost"),
    ({"wind": ("raise", 1)}, "direction must be one of"),
    ({"wind": ("increase", -1)}, "must not be negative"),
    ({"wind": ("increase", "ten")}, "percentage"),
    ({"wind": ("increase", None)}, "number or a percentage"),
    ({"wind": ("increase", 1, 2)}, "direction, delta"),
])
def test_bad_requests(space, request_, match):
    with pytest.raises(ValueError, match=match):
        space.navigate(space.optimum, request_)


def test_start_needs_every_value(space):
    with pytest.raises(ValueError, match="no value for"):
        space.navigate(space.optimum.drop("solar"), {"wind": "increase"})


def test_navigation_table_and_repr(space):
    result = space.navigate(space.optimum, {"wind": ("increase", 3000), "TOTEX": ("keep", 30)})
    table = result.to_frame()
    assert list(table.columns) == ["unit", "start", "target", "change", "request"]
    assert table.loc["wind", "change"] == pytest.approx(2000)
    assert table.loc["wind", "request"] == "increase by 3,000 MW"
    assert table.loc["TOTEX", "request"] == "keep within ±30.00 EUR"
    assert table.loc["solar", "request"] == ""
    assert space.navigate(space.optimum, {"wind": "keep"}).to_frame().loc["wind", "request"] == "keep"
    text = repr(result)
    assert text.startswith("Navigation: 2 requests, 2 adjusted")
    assert "+2,000" in text and "Adjusted:" in text and "Traverse:" in text
    assert "<li>" in result._repr_html_()


def test_navigation_weights_make_the_target(space):
    result = space.navigate(space.optimum, {"wind": ("increase", 1000), "TOTEX": ("keep", 30)})
    assert result.alpha.sum() == pytest.approx(1)
    mixed = space.points.loc[result.alpha.index].mul(result.alpha, axis=0).sum()
    assert mixed.to_numpy() == pytest.approx(result.target.to_numpy())
    assert result.traverse.breakpoints[0].point["wind"] == pytest.approx(2000)


# ---------------------------------------------------------------- outputs


def test_outputs_table(set_dir):
    assert mc.load(set_dir).outputs().to_dict("index") == {
        "capacity": {"unit": "GW", "index": "technology", "description": "capacity per technology"}}
    assert mc.load(set_dir, units="model").outputs().loc["capacity", "unit"] == "MW"


def test_output_of_a_point(space):
    capacity = space.output("capacity", 3)
    assert capacity.name == "capacity"
    assert capacity.attrs["unit"] == "MW"
    assert list(capacity.index) == ["wind", "solar"]
    assert capacity.to_dict() == {"wind": 4000.0, "solar": 100.0}


def test_output_of_a_navigation_is_the_mix_of_its_target(space):
    result = space.navigate(space.optimum, {"wind": ("increase", 1000), "TOTEX": ("keep", 30)})
    capacity = space.output("capacity", result)
    assert capacity["wind"] == pytest.approx(result.target["wind"])  # 3250
    assert capacity["solar"] == pytest.approx(result.target["solar"])  # 250


def test_output_along_the_traverse(space):
    result = space.navigate(space.optimum, {"wind": ("increase", 1000), "TOTEX": ("keep", 30)})
    halfway = space.output("capacity", result.traverse.alpha_at(0.5))
    assert halfway["wind"] == pytest.approx(result.traverse.at(0.5)["wind"])


def test_output_of_poi_values_uses_their_cheapest_mix(space):
    capacity = space.output("capacity", pd.Series({"wind": 3250.0, "solar": 250.0, "TOTEX": 0.0}))
    assert capacity["wind"] == pytest.approx(3250)
    assert capacity["solar"] == pytest.approx(250)


def test_output_in_display_units(set_dir):
    capacity = mc.load(set_dir).output("capacity", 3)
    assert capacity.attrs["unit"] == "GW"
    assert capacity["wind"] == pytest.approx(4.0)


def test_output_errors(space, set_dir):
    with pytest.raises(ValueError, match="no output 'dispatch'"):
        space.output("dispatch", 0)
    with pytest.raises(ValueError, match="not a design of this space"):
        mc.load(set_dir, units="model", slack=0.01).output("capacity", 3)  # found at 5% only
    with pytest.raises(TypeError, match="a design is"):
        space.output("capacity", "optimum")


# ---------------------------------------------------------------- traverse
#
# One PoI, wind, with the convex cost 1000 + max(0.02 (w - 2000), -0.04 (w - 2000))
# EUR, sampled at 1000, 2000, 3000 and 4000 MW, each with a subgradient as its dual.


@pytest.fixture
def line_dir(tmp_path):
    manifest = mga_format.Manifest.model_validate({
        "format": "mga-near-optimal-set", "version": 1, "created": "2026-11-07T12:00:00+00:00",
        "cost": {"name": "TOTEX", "unit": "EUR", "optimum": 1000.0},
        "pois": [{"name": "wind", "unit": "MW", "definition": {}}],
        "slack_levels": [{"slack": 0.05, **LEVEL}],
        "fingerprint": {"network": "x"},
        "display": {"wind": {"unit": "GW", "factor": 1e-3}, "TOTEX": {"unit": "kEUR", "factor": 1e-3}},
    })
    index = pd.Index([0, 1, 2, 3], name="point")
    points = pd.DataFrame({"slack": [0.05, 0.0, 0.05, 0.05], "kind": ["box", "optimum", "box", "box"],
                           "wind": [1000.0, 2000.0, 3000.0, 4000.0],
                           "TOTEX": [1040.0, 1000.0, 1020.0, 1040.0]}, index=index)
    duals = pd.DataFrame({"wind": [-0.04, 0.0, 0.02, 0.02]}, index=index)
    directory = tmp_path / "line"
    mga_format.write(directory, mga_format.NearOptimalSet(manifest, points, duals))
    return directory


def _true_cost(wind):
    return 1000 + max(0.02 * (wind - 2000), -0.04 * (wind - 2000))


def test_traverse_brackets_the_true_cost(line_dir):
    space = mc.load(line_dir, units="model")
    traverse = space.traverse(space.points.loc[0], space.points.loc[3])
    frame = traverse.to_frame()
    assert frame.index[0] == 0 and frame.index[-1] == 1
    assert len(frame) > 2  # bends at the optimum, wind = 2000
    for beta in np.linspace(0, 1, 41):
        design = traverse.at(beta)
        assert design["wind"] == pytest.approx(1000 + 3000 * beta)  # PoIs move in a straight line
        true = _true_cost(design["wind"])
        assert traverse.lower_bound_at(beta) <= true + 1e-6
        assert design["TOTEX"] >= true - 1e-6
    # Tight where the samples are: at the start and the end.
    assert traverse.lower_bound_at(0) == pytest.approx(1040)
    assert traverse.lower_bound_at(1) == pytest.approx(1040)
    assert traverse.lower_bound.min() >= 1000 - 1e-9  # never below the optimum
    gaps = [traverse.cost_at(b) - traverse.lower_bound_at(b) for b in np.linspace(0, 1, 401)]
    assert max(gaps) <= traverse.largest_gap() + 1e-9
    assert traverse.cost_at(0.5) == pytest.approx(traverse.at(0.5)["TOTEX"])


def test_optimum_bounds_from_below(line_dir):
    """From the optimum to 3000 MW, the optimum's dual (0) and 3000 MW's give the
    lines 1000 and 1020 + 0.02 (w - 3000); the constant optimum line is 1000."""
    space = mc.load(line_dir, units="model")
    traverse = space.traverse(space.points.loc[1], space.points.loc[2])
    assert traverse.lower_bound_at(0) == pytest.approx(1000)
    assert traverse.lower_bound_at(1) == pytest.approx(1020)


def test_traverse_in_display_units(line_dir):
    model = mc.load(line_dir, units="model")
    display = mc.load(line_dir)
    assert display.duals.loc[0, "wind"] == pytest.approx(-0.04)  # EUR/MW = kEUR/GW
    assert display.optimal_cost == pytest.approx(1.0)
    shown = display.traverse(display.points.loc[0], display.points.loc[3])
    stored = model.traverse(model.points.loc[0], model.points.loc[3])
    for beta in (0, 0.3, 0.5, 1):
        assert shown.lower_bound_at(beta) == pytest.approx(stored.lower_bound_at(beta) * 1e-3)


def test_alpha_at(line_dir):
    space = mc.load(line_dir, units="model")
    traverse = space.traverse(space.points.loc[0], space.points.loc[3])
    alpha = traverse.alpha_at(0.5)
    assert alpha.sum() == pytest.approx(1)
    assert traverse.alpha_at(0).equals(traverse.breakpoints[0].alpha)
    with pytest.raises(ValueError, match="between 0 and 1"):
        traverse.alpha_at(1.5)


def test_weights_at(line_dir):
    space = mc.load(line_dir, units="model")
    traverse = space.traverse(space.points.loc[0], space.points.loc[3])
    start = traverse.weights_at(0)
    assert list(start.columns) == ["weight", "slack", "kind"]  # the set records no key
    assert list(start.index) == [0] and start.loc[0, "weight"] == pytest.approx(1)
    halfway = traverse.weights_at(0.5)  # wind 2500: half the optimum, half 3000 MW
    assert sorted(halfway.index) == [1, 2]
    assert list(halfway["weight"]) == pytest.approx([0.5, 0.5])
    assert halfway.loc[1, "kind"] == "optimum" and halfway.loc[2, "slack"] == 0.05


def test_weights_at_with_keys(line_dir, tmp_path):
    nos = mga_format.read(line_dir)
    nos.points.insert(2, "key", ["a", None, "b", "c"])
    mga_format.write(tmp_path / "keyed", nos)
    space = mc.load(tmp_path / "keyed", units="model")
    traverse = space.traverse(space.points.loc[0], space.points.loc[3])
    weights = traverse.weights_at(0.9)
    assert list(weights.columns) == ["weight", "slack", "kind", "key"]
    assert weights["weight"].is_monotonic_decreasing
    assert weights["weight"].sum() == pytest.approx(1)
    assert weights["key"].equals(nos.points.loc[weights.index, "key"])
    assert traverse.weights_at(1)["key"].to_dict() == {3: "c"}


def test_traverse_repr_and_plot(line_dir):
    space = mc.load(line_dir, units="model")
    traverse = space.traverse(space.points.loc[0], space.points.loc[3])
    text = repr(traverse)
    assert text.startswith("Traverse: ") and "lower bound" in text
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    ax = traverse.plot()
    labels = [line.get_label() for line in ax.lines]
    assert labels == ["TOTEX (cheapest mix of designs)", "lower bound", "optimal cost", "budget (slack 0.05)"]
    assert ax.lines[3].get_ydata()[0] == pytest.approx(space.optimal_cost * 1.05)
    assert ax.get_ylabel() == "TOTEX [EUR]"
    ax = traverse.plot(beta=0.5)
    marker = ax.lines[-1]
    assert list(marker.get_xdata()) == [0.5]
    assert marker.get_ydata()[0] == pytest.approx(traverse.cost_at(0.5))


def test_plots_of_the_pois(space):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    result = space.navigate(0, {"wind": ("increase", 500)})
    axes = space.plot_ranges()
    assert len(axes) == 1  # wind and solar share a unit
    assert [t.get_text() for t in axes[0].get_xticklabels()] == ["wind", "solar"]
    bars = axes[0].patches
    assert [b.get_y() for b in bars] == [1000.0, 100.0]  # from the smallest value...
    assert [b.get_height() for b in bars] == [3000.0, 800.0]  # ...to the largest

    axes = result.traverse.plot_pois()
    heights = [b.get_height() for b in axes[0].patches]
    assert heights == pytest.approx([*result.start[["wind", "solar"]], *result.target[["wind", "solar"]]])
    assert [t.get_text() for t in axes[0].get_legend().get_texts()] == ["start", "target"]
    axes = result.traverse.plot_pois(beta=0.5)
    assert [b.get_height() for b in axes[0].patches] == pytest.approx(list(result.traverse.at(0.5)[["wind", "solar"]]))
    assert [t.get_text() for t in axes[0].get_legend().get_texts()] == ["β = 0.5", "start", "target"]
    start_lines, target_lines = axes[0].collections
    assert [seg[0][1] for seg in start_lines.get_segments()] == pytest.approx(list(result.start[["wind", "solar"]]))
    assert [seg[0][1] for seg in target_lines.get_segments()] == pytest.approx(list(result.target[["wind", "solar"]]))

    ax = result.traverse.plot_radar()
    start, target = ax.lines[0].get_ydata(), ax.lines[1].get_ydata()
    # wind 2000 of 1000..4000 and solar 500 of 100..900; the first value closes the loop.
    assert list(start) == pytest.approx([1 / 3, 0.5, 1 / 3])
    assert 0 <= min(target) and max(target) <= 1


def test_display_units_split_the_panels(set_dir):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    axes = mc.load(set_dir).plot_ranges()  # wind in GW, solar in MW
    assert [ax.get_ylabel() for ax in axes] == ["GW", "MW"]


def test_plots_of_the_set_use_its_labels_and_colors(set_dir, tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    nos = mga_format.read(set_dir)
    plots = {"capacity": {"type": "bar", "output": "capacity", "x": "technology"}}
    manifest = mga_format.Manifest.model_validate({**nos.manifest.model_dump(), "plots": plots})
    sets = {"technology": pd.DataFrame({"technology": ["wind", "solar"], "label": ["Wind", "Solar"],
                                        "color": ["#0000ff", "#ffff00"]})}
    outputs = {"capacity": mga_format.read_output(set_dir, "capacity")}
    mga_format.write(tmp_path / "styled", mga_format.NearOptimalSet(manifest, nos.points, nos.duals, sets), outputs)
    space = mc.load(tmp_path / "styled", units="model")

    assert list(space.plots().index) == ["capacity"]
    ax = space.plot("capacity", 3)
    assert [b.get_height() for b in ax.patches] == [4000.0, 100.0]  # design 3's wind and solar
    assert [matplotlib.colors.to_hex(b.get_facecolor()) for b in ax.patches] == ["#0000ff", "#ffff00"]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Wind", "Solar"]
    assert ax.get_ylabel() == "capacity [MW]"
    with pytest.raises(ValueError, match="no plot 'missing'"):
        space.plot("missing", 3)
    # A PoI of one component takes its colour; the fixture's PoIs have empty definitions.
    assert [matplotlib.colors.to_hex(b.get_facecolor()) for b in space.plot_ranges()[0].patches] == ["#1f77b4"] * 2
    pois = [poi.model_copy(update={"definition": {"Generator": {"p_nom": {poi.name: 1.0}}}}) for poi in manifest.pois]
    manifest = manifest.model_copy(update={"pois": pois})
    mga_format.write(tmp_path / "defined", mga_format.NearOptimalSet(manifest, nos.points, nos.duals, sets), outputs)
    bars = mc.load(tmp_path / "defined", units="model").plot_ranges()[0].patches
    assert [matplotlib.colors.to_hex(b.get_facecolor()) for b in bars] == ["#0000ff", "#ffff00"]


def test_projection_outlines_the_hull_of_the_designs(space):
    matplotlib = pytest.importorskip("matplotlib")
    pytest.importorskip("scipy")
    matplotlib.use("Agg")
    ax = space.plot_projection("wind", "solar")
    outline = ax.lines[0].get_xydata()
    # (1500, 700) and (2500, 400) lie on the hull's edges, so only three vertices remain.
    assert {tuple(p) for p in outline} == {(1000.0, 900.0), (2000.0, 500.0), (4000.0, 100.0)}
    assert tuple(outline[0]) == tuple(outline[-1])  # closed
    assert len(ax.collections[-1].get_offsets()) == 5  # every design
    assert ax.get_xlabel() == "wind [MW]" and ax.get_ylabel() == "solar [MW]"

    result = space.navigate(0, {"wind": ("increase", 500)})
    ax = space.plot_projection("wind", "TOTEX", result)
    labels = [line.get_label() for line in ax.lines]
    assert labels[-3:] == ["path", "start", "target"]
    assert tuple(ax.lines[-1].get_xydata()[0]) == pytest.approx((result.target["wind"], result.target["TOTEX"]))
    with pytest.raises(ValueError, match="not a PoI or the cost"):
        space.plot_projection("wind", "nuclear")
    with pytest.raises(ValueError, match="must be different"):
        space.plot_projection("wind", "wind")
