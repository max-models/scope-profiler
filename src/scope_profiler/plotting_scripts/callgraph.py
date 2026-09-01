"""Call-graph rendering (PyVis network view of explicit caller/callee edges)."""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from scope_profiler import plotting_scripts as _ps
from scope_profiler.plotting_scripts._utils import (
    DEFAULT_CMAP,
    _add_pyvis_controls,
    _get_cmap_colors,
    _write_csv,
    _write_json,
)
from scope_profiler.results import ProfilingResults


def plot_callgraph(
    profiling_data: ProfilingResults,
    rank: int = 0,
    include=None,
    exclude=None,
    filepath: str | None = None,
    show: bool = False,
    verbose: bool = True,
    cmap: str = DEFAULT_CMAP,
    data_filepath: str | Path | None = None,
    data_format: str = "csv",
    backend: str = "matplotlib",
    return_fig: bool = False,
    compact: bool = False,
    fluid: bool = False,
) -> object | None:
    """Plot the explicit call graph, without using timestamps or durations.

    When ``compact`` is true, all invocations of a region are represented by
    one node named after that region.  Edges then describe distinct
    caller/callee region relationships rather than individual call IDs.
    ``fluid`` applies a deterministic force-directed layout to the compact
    graph, similar to an Obsidian-style node graph.
    """
    if isinstance(profiling_data, Sequence) and not isinstance(
        profiling_data, ProfilingResults
    ):
        if len(profiling_data) != 1:
            raise ValueError("callgraph accepts one profiling file at a time")
        profiling_data = profiling_data[0]
    nodes = profiling_data.call_graph(rank=rank, include=include, exclude=exclude)
    if not nodes:
        raise ValueError("No calls recorded for the requested rank or filters.")
    edges = None
    compact = compact or fluid
    if compact:
        regions_by_name = {
            region.name: region
            for region in profiling_data.get_regions(include=include, exclude=exclude)
        }
        # The call graph is a DAG in call-id order.  Use per-invocation
        # exclusive time as the node weight and dynamic programming to find
        # the longest cumulative root-to-leaf chain.
        exclusive_by_call = {}
        cumulative_by_call = {}
        for node in nodes:
            region = regions_by_name.get(node["name"])
            rank_region = region.regions.get(rank) if region is not None else None
            if rank_region is None or node["call_index"] >= len(
                rank_region.exclusive_durations
            ):
                exclusive = 0.0
            else:
                exclusive = float(rank_region.exclusive_durations[node["call_index"]])
            exclusive_by_call[node["call_id"]] = exclusive
            parent_total = cumulative_by_call.get(node["parent_id"], 0.0)
            cumulative_by_call[node["call_id"]] = exclusive + parent_total
        endpoint = max(cumulative_by_call, key=cumulative_by_call.get, default=None)
        critical_ids = set()
        by_id = {node["call_id"]: node for node in nodes}
        while endpoint is not None and endpoint in by_id:
            critical_ids.add(endpoint)
            endpoint = by_id[endpoint]["parent_id"]
        compact_nodes = []
        seen_names = set()
        for node in nodes:
            if node["name"] not in seen_names:
                seen_names.add(node["name"])
                region = regions_by_name[node["name"]]
                rank_region = region.regions.get(rank)
                calls = rank_region.num_calls if rank_region is not None else 0
                total = rank_region.total_duration if rank_region is not None else 0.0
                exclusive_total = (
                    float(np.sum(rank_region.exclusive_durations))
                    if rank_region is not None
                    else 0.0
                )
                source = ""
                if rank_region is not None and rank_region.has_source:
                    source = f"{rank_region.source_file}:{rank_region.source_lineno}"
                compact_nodes.append(
                    {
                        "name": node["name"],
                        "depth": node["depth"],
                        "calls": calls,
                        "total_duration": total,
                        "exclusive_duration": exclusive_total,
                        "average_duration": total / calls if calls else 0.0,
                        "source": source,
                        # The node's hover box, from the region's own
                        # get_summary() rather than a second set of numbers
                        # assembled here.
                        "summary": (
                            _ps._hover_summary(rank_region, title=node["name"])
                            if rank_region is not None
                            else f"<b>{node['name']}</b>"
                        ),
                        "critical": any(
                            candidate["name"] == node["name"]
                            for candidate in nodes
                            if candidate["call_id"] in critical_ids
                        ),
                    }
                )
        edge_counts = {}
        by_id = {node["call_id"]: node for node in nodes}
        for node in nodes:
            parent = by_id.get(node["parent_id"])
            if parent is not None:
                edge = (parent["name"], node["name"])
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        edges = sorted(edge_counts)
        nodes = compact_nodes

    def fluid_positions():
        """Return a small deterministic force-directed layout for the graph."""
        names = [node["name"] for node in nodes]
        if not fluid or len(names) < 2:
            return None
        index = {name: i for i, name in enumerate(names)}
        points = np.column_stack(
            (
                np.cos(np.linspace(0, 2 * np.pi, len(names), endpoint=False)),
                np.sin(np.linspace(0, 2 * np.pi, len(names), endpoint=False)),
            )
        ).astype(float)
        graph_edges = [(index[parent], index[child]) for parent, child in edges]
        for step in range(120):
            delta = points[:, None, :] - points[None, :, :]
            distance = np.maximum(np.linalg.norm(delta, axis=2), 1e-3)
            force = (delta / distance[:, :, None] ** 2).sum(axis=1)
            for left, right in graph_edges:
                vector = points[right] - points[left]
                distance_edge = max(float(np.linalg.norm(vector)), 1e-3)
                pull = vector * (distance_edge - 0.8) / distance_edge
                force[left] += pull
                force[right] -= pull
            temperature = 0.08 * (1.0 - step / 120.0)
            points += np.clip(force, -temperature, temperature)
        points -= points.mean(axis=0)
        scale = max(float(np.abs(points).max()), 1e-6)
        points /= scale
        return {name: tuple(point) for name, point in zip(names, points)}

    fluid_layout = fluid_positions()

    if backend == "pyvis":
        try:
            from pyvis.network import Network
        except ImportError as exc:
            raise ImportError(
                "pyvis is required for the pyvis callgraph backend. "
                "Install scope-profiler[graph]."
            ) from exc
        graph = Network(height="800px", width="100%", directed=True)
        key = (lambda node: node["name"]) if compact else (lambda node: node["call_id"])
        node_keys = {key(node) for node in nodes}
        max_total = max(
            (node.get("exclusive_duration", 0.0) for node in nodes), default=0.0
        )
        for node in nodes:
            node_key = key(node)
            label = node["name"] if compact else f"{node['name']} (#{node['call_id']})"
            if compact:
                size = (
                    16 + 18 * (node["exclusive_duration"] / max_total) ** 0.5
                    if max_total
                    else 18
                )
                intensity = node["exclusive_duration"] / max_total if max_total else 0.0
                red = int(224 - 130 * intensity)
                green = int(242 - 150 * intensity)
                blue = int(255 - 25 * intensity)
                background = f"#{red:02x}{green:02x}{blue:02x}"
                border = "#dc2626" if node["critical"] else "#64748b"
                title = (
                    node["summary"]
                    + ("<br><b>Critical path</b>" if node["critical"] else "")
                    + (f"<br>Source: {node['source']}" if node["source"] else "")
                )
                graph.add_node(
                    node_key,
                    label=label,
                    title=title,
                    level=node["depth"],
                    size=size,
                    borderWidth=4 if node["critical"] else 1,
                    color={
                        "background": background,
                        "border": border,
                        "highlight": {"background": "#fef08a", "border": "#b91c1c"},
                    },
                )
            else:
                graph.add_node(
                    node_key, label=label, title=node["name"], level=node["depth"]
                )
        graph_edges = (
            edges
            if compact
            else [
                (node["parent_id"], node["call_id"])
                for node in nodes
                if node["parent_id"] is not None
            ]
        )
        for parent, child in graph_edges:
            if parent in node_keys and child in node_keys:
                count = edge_counts.get((parent, child), 1) if compact else 1
                critical_edge = compact and all(
                    node["critical"]
                    for node in nodes
                    if node["name"] in {parent, child}
                )
                graph.add_edge(
                    parent,
                    child,
                    arrows="to",
                    label=f"×{count}" if compact and count > 1 else "",
                    title=f"{count} calls" if compact else "",
                    smooth=(
                        {"type": "curvedCW"} if parent == child else {"type": "dynamic"}
                    ),
                    color="#dc2626" if critical_edge else "#94a3b8",
                    width=3 if critical_edge else 1,
                )
        # Keep the call-depth structure legible while allowing nodes on the
        # same level to spread and settle horizontally like an Obsidian graph.
        graph.set_options(
            json.dumps(
                {
                    "layout": {
                        "hierarchical": {
                            "enabled": True,
                            "direction": "UD",
                            "sortMethod": "directed",
                            "levelSeparation": 140,
                            "nodeSpacing": 180,
                            "treeSpacing": 220,
                        }
                    },
                    "physics": {
                        "enabled": True,
                        "solver": "hierarchicalRepulsion",
                        "hierarchicalRepulsion": {
                            "nodeDistance": 180,
                            "centralGravity": 0.1,
                            "springLength": 140,
                            "springConstant": 0.01,
                            "avoidOverlap": 1,
                        },
                        "stabilization": {"iterations": 250},
                    },
                }
            )
        )
        if filepath:
            output_path = Path(filepath)
        elif show:
            import tempfile

            output_path = Path(tempfile.mkdtemp()) / "callgraph.html"
        else:
            output_path = None
        if output_path is not None:
            graph.write_html(str(output_path), open_browser=False)
            _add_pyvis_controls(output_path)
            if show:
                import webbrowser

                webbrowser.open(output_path.resolve().as_uri())
        return graph if return_fig else None
    if data_filepath:
        if compact:
            if data_format == "json":
                _write_json(
                    data_filepath,
                    {
                        "regions": nodes,
                        "edges": [
                            {"parent": parent, "child": child}
                            for parent, child in edges
                        ],
                    },
                )
            else:
                _write_csv(data_filepath, ["parent", "child"], edges)
        else:
            rows = [
                [node["call_id"], node["parent_id"], node["name"], node["depth"]]
                for node in nodes
            ]
            if data_format == "json":
                _write_json(
                    data_filepath,
                    {
                        "calls": [
                            dict(zip(("call_id", "parent_id", "name", "depth"), row))
                            for row in rows
                        ]
                    },
                )
            else:
                _write_csv(
                    data_filepath, ["call_id", "parent_id", "name", "depth"], rows
                )

    if backend == "plotly":
        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise ImportError("plotly is required for the callgraph plot") from exc
        figure = go.Figure()
        key = (lambda node: node["name"]) if compact else (lambda node: node["call_id"])
        positions = fluid_layout or {
            key(node): (index, -node["depth"]) for index, node in enumerate(nodes)
        }
        graph_edges = (
            edges
            if compact
            else [
                (node["parent_id"], node["call_id"])
                for node in nodes
                if node["parent_id"] is not None
            ]
        )
        for parent, child in graph_edges:
            if parent in positions and child in positions:
                x0, y0 = positions[parent]
                x1, y1 = positions[child]
                figure.add_trace(
                    go.Scatter(
                        x=[x0, x1],
                        y=[y0, y1],
                        mode="lines",
                        line={"color": "#999"},
                        showlegend=False,
                    )
                )
        figure.add_trace(
            go.Scatter(
                x=[positions[key(node)][0] for node in nodes],
                y=[positions[key(node)][1] for node in nodes],
                text=[
                    node["name"] if compact else f"{node['name']} ({node['call_id']})"
                    for node in nodes
                ],
                mode="markers+text",
                textposition="bottom center",
                showlegend=False,
            )
        )
        figure.update_layout(
            title=f"Call graph (rank {rank})", xaxis_visible=False, yaxis_visible=False
        )
        if filepath:
            figure.write_html(str(filepath))
        if show:
            figure.show()
        return figure if return_fig else None

    import matplotlib.pyplot as plt

    key = (lambda node: node["name"]) if compact else (lambda node: node["call_id"])
    positions = fluid_layout or {
        key(node): (index, -node["depth"]) for index, node in enumerate(nodes)
    }
    fig, axis = plt.subplots(
        figsize=(
            max(8, len(nodes) * 0.8),
            max(3, 2 + max(node["depth"] for node in nodes)),
        )
    )
    graph_edges = (
        edges
        if compact
        else [
            (node["parent_id"], node["call_id"])
            for node in nodes
            if node["parent_id"] is not None
        ]
    )
    for parent, child in graph_edges:
        if parent in positions and child in positions:
            x0, y0 = positions[parent]
            x1, y1 = positions[child]
            axis.plot([x0, x1], [y0, y1], color="#999999", linewidth=1, zorder=1)
    colors = _get_cmap_colors(cmap, max(1, len({node["name"] for node in nodes})))
    color_by_name = {
        name: colors[index % len(colors)]
        for index, name in enumerate(sorted({node["name"] for node in nodes}))
    }
    axis.scatter(
        [positions[key(node)][0] for node in nodes],
        [positions[key(node)][1] for node in nodes],
        c=[color_by_name[node["name"]] for node in nodes],
        s=140,
        zorder=2,
    )
    for node in nodes:
        x, y = positions[key(node)]
        label = node["name"] if compact else f"{node['name']}\n#{node['call_id']}"
        axis.text(x, y - 0.12, label, ha="center", va="top", fontsize=8)
    axis.set_title(f"Call graph (rank {rank})")
    axis.set_axis_off()
    fig.tight_layout()
    if filepath:
        fig.savefig(filepath, bbox_inches="tight")
    if show:
        plt.show()
    return (fig, axis) if return_fig else None
