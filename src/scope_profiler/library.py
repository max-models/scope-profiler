"""Opt-in instrumentation for libraries: one tag, a name prefix, off by default."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from typing import Any

from scope_profiler.profile_manager import ProfileManager


class LibraryProfiler:
    """The ``profile`` decorator and ``region`` of one library's opt-in tag.

    Created by :func:`library`. Every region it makes is named
    ``"<tag>.<name>"`` and carries ``tag`` (plus any extra ``tags``), so it
    records nothing unless the application enables the tag.

    Parameters
    ----------
    tag : str
        The library's opt-in tag, also the prefix of its region names.

    Attributes
    ----------
    tag : str
        The opt-in tag.
    tags : tuple of str
        ``(tag,)``, the tags every region of the library carries.

    Examples
    --------
    >>> import scope_profiler as sp
    >>> prof = sp.library("mylib")
    >>> prof.name("solve")
    'mylib.solve'
    """

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.tags = (tag,)

    def name(self, region_name: str) -> str:
        """Return the full name of the library's region ``region_name``.

        Parameters
        ----------
        region_name : str
            The name without the tag prefix.

        Returns
        -------
        str
            ``"<tag>.<region_name>"``.
        """
        return f"{self.tag}.{region_name}"

    def _tags(self, tags: Iterable[str] | None) -> tuple[str, ...]:
        extra = () if tags is None else tuple(t for t in tags if t != self.tag)
        return (self.tag, *extra)

    def profile(
        self,
        region_name: str | Callable | None = None,
        *,
        tags: Iterable[str] | None = None,
        **kwargs: Any,
    ):
        """Profile a function as the library region ``"<tag>.<region_name>"``.

        :meth:`ProfileManager.profile` with the name prefixed and the tag
        added. Works with and without parentheses; without a name the region
        is ``"<tag>.<qualified name of the function>"``, e.g.
        ``"mylib.Solver.step"`` for a method.

        Parameters
        ----------
        region_name : str or callable, optional
            The name without the tag prefix (or the function itself, when
            used without parentheses).
        tags : iterable of str, optional
            Further tags, besides the library's own.
        **kwargs
            ``recursive``, ``metadata`` and ``record_outcome``, passed on to
            :meth:`ProfileManager.profile`.

        Returns
        -------
        callable
            The decorated function, or a decorator.
        """
        if callable(region_name):
            return self.profile(tags=tags, **kwargs)(region_name)

        def decorator(func):
            name = self.name(region_name or func.__qualname__)
            return ProfileManager.profile(name, tags=self._tags(tags), **kwargs)(func)

        return decorator

    def region(
        self,
        region_name: str,
        *,
        tags: Iterable[str] | None = None,
        functions=None,
    ):
        """Return the library region ``"<tag>.<region_name>"`` as a context manager.

        :meth:`ProfileManager.region` with the name prefixed and the tag
        added.

        Parameters
        ----------
        region_name : str
            The name without the tag prefix.
        tags : iterable of str, optional
            Further tags, besides the library's own.
        functions : list of callable, optional
            Functions to line-profile, as for :meth:`ProfileManager.region`.

        Returns
        -------
        context manager
            The region.
        """
        return ProfileManager.region(
            self.name(region_name), functions=functions, tags=self._tags(tags)
        )

    def __repr__(self) -> str:
        return f"LibraryProfiler(tag={self.tag!r})"


def library(tag: str) -> LibraryProfiler:
    """Register a library's opt-in tag and return its profiler.

    The one call a library makes to ship its own instrumentation, off unless
    the application enables ``tag`` (``setup(enable_tags=[tag])``,
    ``SCOPE_PROFILER_ENABLE_TAGS``, ``scope-profiler run --enable-tag``)::

        # mylib/_profiling.py
        import scope_profiler as sp

        prof = sp.library("mylib")

        # mylib/solver.py
        from mylib._profiling import prof

        @prof.profile("solve")  # the region "mylib.solve", tagged "mylib"
        def solve(): ...

        with prof.region("assemble"):  # the region "mylib.assemble"
            ...

    It registers ``tag`` with :meth:`ProfileManager.register_opt_in_tag`
    (recording the calling module for ``scope-profiler tags``). For the
    command to find the tag without importing the library first, declare an
    entry point in ``pyproject.toml``::

        [project.entry-points."scope_profiler.opt_in_tags"]
        mylib = "mylib"

    :mod:`scope_profiler.testing` checks all of this in one test.

    Parameters
    ----------
    tag : str
        The opt-in tag, conventionally the import name of the library; also
        the prefix of its region names.

    Returns
    -------
    LibraryProfiler
        The library's ``profile`` decorator and ``region``.
    """
    module = sys._getframe(1).f_globals.get("__name__")
    ProfileManager._register_opt_in_tags((tag,), module)
    return LibraryProfiler(tag)
