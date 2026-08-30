"""Tests for ``scope_profiler.mcp_server.server``.

Exercises the actual MCP server object (tool registration, JSON schemas,
argument validation, dispatch and error handling) without spawning a real
transport (stdio/SSE) or talking to an external LLM/Claude Code -- FastMCP's
``list_tools()``/``call_tool()`` run the same code path in-process.

Skipped entirely if the optional ``mcp`` extra is not installed, since it is
not a dependency of the base package (``pip install scope-profiler`` alone
must not require it).
"""

import asyncio
import json

import h5py
import numpy as np
import pytest

mcp = pytest.importorskip("mcp", reason="the optional 'mcp' extra is not installed")

from scope_profiler.mcp_server.server import create_server

NS = 1_000_000_000

EXPECTED_TOOLS = {
    "inspect_profile",
    "compare_profiles",
    "run_profile",
    "plot_profile",
    "run_benchmark",
    "compare_benchmarks",
}


def _write_sample_h5(path, rank_regions, metadata=None):
    with h5py.File(path, "w") as h5file:
        meta_grp = h5file.create_group("metadata")
        for key, value in (metadata or {}).items():
            meta_grp.attrs[key] = value
        for rank, regions in rank_regions.items():
            regions_group = h5file.create_group(f"rank{rank}").create_group("regions")
            for region_name, payload in regions.items():
                region_group = regions_group.create_group(region_name)
                starts, ends = payload
                region_group.create_dataset(
                    "start_times", data=np.asarray(starts, dtype=np.int64)
                )
                region_group.create_dataset(
                    "end_times", data=np.asarray(ends, dtype=np.int64)
                )


@pytest.fixture
def sample_file(tmp_path):
    path = tmp_path / "run.h5"
    _write_sample_h5(
        path,
        {0: {"setup": ([0], [1 * NS]), "solve": ([1 * NS, 4 * NS], [3 * NS, 6 * NS])}},
        metadata={"start_time_ns": 0, "finalize_time_ns": 6 * NS},
    )
    return path


def _run(coro):
    return asyncio.run(coro)


def _tool_json(result) -> dict:
    """Unwrap a successful ``call_tool`` result's single TextContent as JSON."""
    (block,) = result
    return json.loads(block.text)


class TestServerCreation:
    def test_create_server_returns_a_fastmcp_instance(self):
        from mcp.server.fastmcp import FastMCP

        server = create_server()
        assert isinstance(server, FastMCP)
        assert server.name == "scope-profiler"

    def test_creating_the_server_twice_gives_independent_instances(self):
        assert create_server() is not create_server()


class TestToolRegistration:
    def test_all_tools_are_registered(self):
        server = create_server()
        tools = _run(server.list_tools())

        assert {tool.name for tool in tools} == EXPECTED_TOOLS

    def test_every_tool_has_a_non_trivial_description(self):
        server = create_server()
        tools = _run(server.list_tools())

        for tool in tools:
            assert tool.description and len(tool.description) > 40

    def test_inspect_profile_schema_requires_only_file_path(self):
        server = create_server()
        tools = _run(server.list_tools())
        by_name = {tool.name: tool for tool in tools}

        schema = by_name["inspect_profile"].inputSchema
        assert schema["required"] == ["file_path"]
        assert "top_n" in schema["properties"]

    def test_compare_profiles_schema_requires_both_paths(self):
        server = create_server()
        tools = _run(server.list_tools())
        by_name = {tool.name: tool for tool in tools}

        schema = by_name["compare_profiles"].inputSchema
        assert set(schema["required"]) == {"baseline_path", "candidate_path"}


class TestArgumentValidation:
    def test_missing_required_argument_is_rejected(self):
        server = create_server()

        with pytest.raises(Exception):  # noqa: B017 - FastMCP validation error type
            _run(server.call_tool("inspect_profile", {}))

    def test_unknown_tool_name_is_rejected(self):
        server = create_server()

        with pytest.raises(Exception):  # noqa: B017
            _run(server.call_tool("not_a_real_tool", {}))


class TestInspectProfileTool:
    def test_returns_structured_content(self, sample_file):
        server = create_server()

        result = _run(
            server.call_tool("inspect_profile", {"file_path": str(sample_file)})
        )
        payload = _tool_json(result)

        assert payload["num_ranks"] == 1
        assert payload["total_time_seconds"] == pytest.approx(6.0)
        assert {r["name"] for r in payload["regions"]["items"]} == {"setup", "solve"}

    def test_missing_file_becomes_a_tool_error(self, tmp_path):
        server = create_server()

        with pytest.raises(Exception, match="not found"):
            _run(
                server.call_tool(
                    "inspect_profile", {"file_path": str(tmp_path / "missing.h5")}
                )
            )


class TestCompareProfilesTool:
    def test_reports_speedup(self, tmp_path, sample_file):
        candidate = tmp_path / "candidate.h5"
        _write_sample_h5(
            candidate,
            {
                0: {
                    "setup": ([0], [1 * NS]),
                    "solve": ([1 * NS, 2 * NS], [2 * NS, 3 * NS]),
                }
            },
            metadata={"start_time_ns": 0, "finalize_time_ns": 3 * NS},
        )

        server = create_server()
        result = _run(
            server.call_tool(
                "compare_profiles",
                {"baseline_path": str(sample_file), "candidate_path": str(candidate)},
            )
        )
        payload = _tool_json(result)

        assert payload["overall"]["faster"] is True
        assert payload["overall"]["speedup"] == pytest.approx(2.0)
