"""Opt-in pytest integration for recording test-phase durations.

The plugin deliberately owns an independent :class:`ProfileManager`.  Tests
are free to set up and finalize the process-wide ``ProfileManager`` themselves
without corrupting the profile pytest is collecting.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scope_profiler.profile_manager import ProfileManager

_MANAGER_ATTRIBUTE = "_scope_profiler_pytest_manager"


def pytest_addoption(parser) -> None:
    """Register the command-line switches without enabling profiling by default."""
    group = parser.getgroup("scope-profiler")
    group.addoption(
        "--scope-profile",
        action="store_true",
        help="record selected pytest tests with scope-profiler",
    )
    group.addoption(
        "--scope-profile-out",
        default="pytest-profile.h5",
        metavar="PATH",
        help="HDF5 output path (default: pytest-profile.h5)",
    )
    group.addoption(
        "--scope-profile-config",
        metavar="FILE",
        help="TOML profiling configuration to apply",
    )
    group.addoption(
        "--scope-profile-phases",
        choices=("call", "all"),
        default="call",
        help="profile test call only, or setup/call/teardown (default: call)",
    )


def _enabled(config) -> bool:
    return bool(config.getoption("--scope-profile"))


def _manager(config) -> ProfileManager:
    return getattr(config, _MANAGER_ATTRIBUTE)


def _region(item, phase: str):
    return _manager(item.config).profile_region(f"pytest::{item.nodeid}::{phase}")


def pytest_sessionstart(session) -> None:
    """Start one independent profiling run for this pytest invocation."""
    config = session.config
    if not _enabled(config):
        return
    if config.getoption("numprocesses", default=0):
        raise pytest.UsageError(
            "--scope-profile does not support pytest-xdist yet; "
            "run without -n so exactly one process writes the HDF5 profile"
        )

    manager = ProfileManager()
    setattr(config, _MANAGER_ATTRIBUTE, manager)
    manager.setup(
        file_path=str(Path(config.getoption("--scope-profile-out"))),
        label="pytest",
        config_path=config.getoption("--scope-profile-config"),
    )


def _profile_phase(item, phase: str) -> bool:
    return _enabled(item.config) and (
        phase == "call" or item.config.getoption("--scope-profile-phases") == "all"
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    if not _profile_phase(item, "setup"):
        yield
        return
    with _region(item, "setup"):
        yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    if not _profile_phase(item, "call"):
        yield
        return
    with _region(item, "call"):
        yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    if not _profile_phase(item, "teardown"):
        yield
        return
    with _region(item, "teardown"):
        yield


def pytest_sessionfinish(session, exitstatus) -> None:
    """Publish the profile even when one or more tests failed."""
    config = session.config
    if not _enabled(config):
        return

    manager = _manager(config)
    manager.finalize(verbose=False)
    terminal = config.pluginmanager.get_plugin("terminalreporter")
    if terminal is not None:
        terminal.write_line(
            "scope-profiler: wrote " + config.getoption("--scope-profile-out")
        )
