"""MCP server exposing scope-profiler to AI coding agents.

Requires the optional ``mcp`` extra (``pip install "scope-profiler[mcp]"``);
importing this package elsewhere is safe, but :mod:`scope_profiler.mcp_server.server`
raises ``ImportError`` without it. The rest of scope-profiler has no
dependency on this package or on ``mcp``.
"""
