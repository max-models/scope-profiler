"""Call-graph rendering across its three backends and both node modes.

The plot is driven by an actual nested profiling run rather than a hand-built
result set, so the call-id/parent-id relationships under test are the ones the
profiler really records.
"""

import csv
import importlib.util
import json

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from scope_profiler import ProfileManager  # noqa: E402
from scope_profiler.plotting_scripts.callgraph import plot_callgraph  # noqa: E402


@pytest.fixture(autouse=True)
def _reset():
    yield
    ProfileManager._reset()
    plt.close("all")


@pytest.fixture
def results(tmp_path):
    """A nested run: one root, two children, one of them called repeatedly."""

    @ProfileManager.profile("leaf")
    def leaf():
        return 1

    @ProfileManager.profile("middle")
    def middle():
        return sum(leaf() for _ in range(3))

    @ProfileManager.profile("sibling")
    def sibling():
        return leaf()

    @ProfileManager.profile("root")
    def root():
        return middle() + sibling()

    # setup()/finalize() rather than session(), so the graph holds only the
    # user's own regions and not the session's own enclosing region.
    ProfileManager.setup(
        file_path=str(tmp_path / "profiling_data.h5"), deactivate_file_output=True
    )
    root()
    return ProfileManager.finalize(verbose=False, return_results=True)


def test_matplotlib_backend_draws_one_marker_per_call(results, tmp_path):
    out = tmp_path / "callgraph.png"
    figure, axis = plot_callgraph(
        results, filepath=str(out), verbose=False, return_fig=True
    )

    # One node per recorded call: root, middle, three leaves, sibling, leaf.
    assert len(results.call_graph()) == 7
    offsets = axis.collections[0].get_offsets()
    assert len(offsets) == 7
    assert out.exists()
    assert figure is not None


def test_matplotlib_backend_returns_nothing_unless_asked(results):
    assert plot_callgraph(results, verbose=False) is None


def test_compact_mode_collapses_calls_into_one_node_per_region(results, tmp_path):
    _, axis = plot_callgraph(
        results,
        compact=True,
        filepath=str(tmp_path / "compact.png"),
        verbose=False,
        return_fig=True,
    )

    # Four distinct region names, however many times each was called.
    assert len(axis.collections[0].get_offsets()) == 4
    labels = {text.get_text() for text in axis.texts}
    assert labels == {"root", "middle", "leaf", "sibling"}


def test_fluid_layout_implies_compact_and_places_nodes_in_the_unit_square(
    results, tmp_path
):
    _, axis = plot_callgraph(
        results,
        fluid=True,
        filepath=str(tmp_path / "fluid.png"),
        verbose=False,
        return_fig=True,
    )

    offsets = axis.collections[0].get_offsets()
    assert len(offsets) == 4
    # The layout is normalised onto [-1, 1] and centred.
    assert abs(offsets).max() <= 1.0 + 1e-9


def test_fluid_layout_is_deterministic(results, tmp_path):
    first = plot_callgraph(results, fluid=True, verbose=False, return_fig=True)[1]
    second = plot_callgraph(results, fluid=True, verbose=False, return_fig=True)[1]

    assert (
        first.collections[0].get_offsets() == second.collections[0].get_offsets()
    ).all()


def test_plotly_backend_builds_traces_for_edges_and_nodes(results, tmp_path):
    out = tmp_path / "callgraph.html"
    figure = plot_callgraph(
        results,
        backend="plotly",
        filepath=str(out),
        verbose=False,
        return_fig=True,
    )

    assert out.exists()
    # One line trace per edge plus the single marker trace.
    assert len(figure.data) == 6 + 1
    assert figure.layout.title.text == "Call graph (rank 0)"


def test_plotly_backend_in_compact_mode_labels_regions(results):
    figure = plot_callgraph(
        results, backend="plotly", compact=True, verbose=False, return_fig=True
    )

    markers = figure.data[-1]
    assert set(markers.text) == {"root", "middle", "leaf", "sibling"}


# The pyvis backend is an optional extra (scope-profiler[graph]). CI installs
# it via the dev extra, so these run there; a partial install skips them
# instead of reporting the missing dependency as a failure.
pyvis_backend = pytest.mark.skipif(
    importlib.util.find_spec("pyvis") is None,
    reason="the optional 'graph' extra (pyvis) is not installed",
)


@pyvis_backend
def test_pyvis_backend_writes_a_document_with_the_added_controls(results, tmp_path):
    out = tmp_path / "callgraph.html"
    graph = plot_callgraph(
        results, backend="pyvis", filepath=str(out), verbose=False, return_fig=True
    )

    assert len(graph.nodes) == 7
    document = out.read_text(encoding="utf-8")
    assert "scope-profiler-controls" in document


@pyvis_backend
def test_pyvis_backend_marks_the_critical_path_in_compact_mode(results, tmp_path):
    graph = plot_callgraph(
        results,
        backend="pyvis",
        compact=True,
        filepath=str(tmp_path / "compact.html"),
        verbose=False,
        return_fig=True,
    )

    assert len(graph.nodes) == 4
    # The heaviest root-to-leaf chain is drawn with the wider red border.
    assert any(node["borderWidth"] == 4 for node in graph.nodes)
    # Repeated caller/callee pairs carry their multiplicity on the edge.
    assert any(edge["label"] == "×3" for edge in graph.edges)


def test_pyvis_backend_reports_a_missing_install(results, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "pyvis.network", None)
    with pytest.raises(ImportError, match="pyvis is required"):
        plot_callgraph(results, backend="pyvis", verbose=False)


def test_data_export_writes_one_csv_row_per_call(results, tmp_path):
    data = tmp_path / "calls.csv"
    plot_callgraph(results, data_filepath=str(data), verbose=False)

    with open(data, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["call_id", "parent_id", "name", "depth"]
    assert len(rows) == 1 + 7


def test_data_export_writes_json_calls(results, tmp_path):
    data = tmp_path / "calls.json"
    plot_callgraph(results, data_filepath=str(data), data_format="json", verbose=False)

    payload = json.loads(data.read_text(encoding="utf-8"))
    assert len(payload["calls"]) == 7
    assert payload["calls"][0]["name"] == "root"
    assert payload["calls"][0]["parent_id"] is None


def test_compact_data_export_writes_edges(results, tmp_path):
    data = tmp_path / "edges.csv"
    plot_callgraph(results, compact=True, data_filepath=str(data), verbose=False)

    with open(data, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["parent", "child"]
    assert ["root", "middle"] in rows
    assert ["middle", "leaf"] in rows


def test_compact_json_export_writes_edges(results, tmp_path):
    data = tmp_path / "edges.json"
    plot_callgraph(
        results,
        compact=True,
        data_filepath=str(data),
        data_format="json",
        verbose=False,
    )

    payload = json.loads(data.read_text(encoding="utf-8"))
    assert {"parent": "root", "child": "sibling"} in payload["edges"]
    assert len(payload["regions"]) == 4


def test_a_single_element_sequence_is_unwrapped(results):
    assert plot_callgraph([results], verbose=False) is None


def test_more_than_one_profile_is_rejected(results):
    with pytest.raises(ValueError, match="one profiling file at a time"):
        plot_callgraph([results, results], verbose=False)


def test_a_rank_without_calls_is_rejected(results):
    with pytest.raises(ValueError, match="No calls recorded"):
        plot_callgraph(results, rank=7, verbose=False)


def test_filters_narrow_the_graph(results):
    _, axis = plot_callgraph(
        results, exclude=["leaf"], compact=True, verbose=False, return_fig=True
    )

    labels = {text.get_text() for text in axis.texts}
    assert "leaf" not in labels


@pyvis_backend
def test_pyvis_show_writes_to_a_temporary_file_and_opens_it(results, monkeypatch):
    """Without a filepath, --show still needs a document on disk to open."""
    import webbrowser

    opened = []
    monkeypatch.setattr(webbrowser, "open", opened.append)

    graph = plot_callgraph(
        results, backend="pyvis", show=True, verbose=False, return_fig=True
    )

    assert len(graph.nodes) == 7
    assert len(opened) == 1
    assert opened[0].startswith("file://")
    assert opened[0].endswith("callgraph.html")


@pyvis_backend
def test_pyvis_without_filepath_or_show_writes_nothing(results, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plot_callgraph(results, backend="pyvis", verbose=False)

    assert list(tmp_path.iterdir()) == []


def test_plotly_show_displays_the_figure(results, monkeypatch):
    shown = []
    monkeypatch.setattr(
        "plotly.graph_objects.Figure.show", lambda self, *a, **k: shown.append(self)
    )

    plot_callgraph(results, backend="plotly", show=True, verbose=False)

    assert len(shown) == 1


def test_matplotlib_show_displays_the_figure(results, monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(True))

    plot_callgraph(results, show=True, verbose=False)

    assert shown == [True]


def test_plotly_backend_reports_a_missing_install(results, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)
    with pytest.raises(ImportError, match="plotly is required"):
        plot_callgraph(results, backend="plotly", verbose=False)
