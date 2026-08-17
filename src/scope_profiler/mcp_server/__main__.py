"""Allows ``python -m scope_profiler.mcp_server`` as an alternative to the
``scope-profiler-mcp`` console script."""

from scope_profiler.mcp_server.server import main

if __name__ == "__main__":
    main()
