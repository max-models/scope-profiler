"""Pick the right reader or writer for a profile path.

scope-profiler writes and reads a run in three shapes: the HDF5 file that is
the default output, the JSON document of :mod:`scope_profiler.json_export`,
and the standalone HTML report of :mod:`scope_profiler.html_report` (which is
write-only -- a report is a rendering of a run, not a copy of it).

Everything that takes a profile path from a user goes through
:func:`read_profile` rather than :func:`~scope_profiler.h5reader.read_h5`, so
a ``.json`` file works anywhere a ``.h5`` file does. The format is chosen by
the file name, the same rule ``scope-profiler run -o`` follows.
"""

from __future__ import annotations

from pathlib import Path

from scope_profiler.json_export import is_json_path
from scope_profiler.results import ProfilingResults

#: Names ending in one of these are HTML reports.
HTML_SUFFIXES = (".html", ".htm")

FORMAT_HDF5 = "hdf5"
FORMAT_JSON = "json"
FORMAT_HTML = "html"


def profile_format(path) -> str:
    """Which of the three formats ``path`` names.

    Anything that is not recognisably JSON or HTML is HDF5: that is the
    default output format, and it is written under whatever name the user
    asked for, extension or not.
    """
    if is_json_path(path):
        return FORMAT_JSON
    if Path(path).name.lower().endswith(HTML_SUFFIXES):
        return FORMAT_HTML
    return FORMAT_HDF5


def read_profile(file_path, verbose: bool = False) -> ProfilingResults:
    """Read a profile from HDF5 or JSON, whichever the name says it is."""
    if profile_format(file_path) == FORMAT_JSON:
        from scope_profiler.json_export import read_json

        return read_json(file_path, verbose=verbose)
    from scope_profiler.h5reader import read_h5

    return read_h5(file_path, verbose=verbose)


def read_profile_summary(file_path, **kwargs) -> ProfilingResults:
    """Read a profile, skipping per-call timestamps where the format can.

    Only HDF5 can be read partially: its per-call columns are separate
    datasets, so a summary-only read never touches them. A JSON document is
    parsed as a whole either way, so this falls back to the full read for one
    -- the caller gets a result set that answers strictly more, never less.
    """
    if profile_format(file_path) == FORMAT_JSON:
        from scope_profiler.json_export import read_json

        return read_json(file_path)
    from scope_profiler.h5reader import read_h5_summary

    return read_h5_summary(file_path, **kwargs)


def write_profile(results, file_path, **kwargs) -> Path:
    """Write ``results`` to ``file_path`` in the format its name asks for.

    Parameters
    ----------
    results : ProfilingResults
        The run to write.
    file_path : str | Path
        Destination. ``.json``/``.json.gz`` writes a JSON profile,
        ``.html`` an HTML report, and anything else an HDF5 file.
    **kwargs
        Passed to the format's own writer.

    Returns
    -------
    Path
        The file written.
    """
    kind = profile_format(file_path)
    if kind == FORMAT_JSON:
        from scope_profiler.json_export import write_json

        return write_json(results, file_path, **kwargs)
    if kind == FORMAT_HTML:
        from scope_profiler.html_report import create_html_report

        return create_html_report(results, file_path, **kwargs)
    from scope_profiler.native_trace import write_results

    return write_results(results, file_path, **kwargs)
