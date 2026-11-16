"""MCP server exposing scope-profiler to AI coding agents.

Requires the optional ``mcp`` extra (``pip install "scope-profiler[mcp]"``);
importing this package elsewhere is safe, but :mod:`scope_profiler.mcp_server.server`
raises ``ImportError`` without it. The rest of scope-profiler does not depend
on ``mcp``; the ``--json`` output of ``inspect``, ``diff`` and ``check`` reuses
the payloads of :mod:`scope_profiler.mcp_server.tools`, which needs no ``mcp``.
"""
