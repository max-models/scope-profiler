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


#: The first bytes of an HDF5 file (the format's own signature).
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"

#: The first bytes of a gzip member, which is how ``.json.gz`` starts.
_GZIP_MAGIC = b"\x1f\x8b"


def sniff_profile_format(path) -> str:
    """Which format the *contents* of ``path`` are in.

    :func:`profile_format` goes by the file name, which is all that can be
    known before a file exists. For a file that does exist the contents are
    the better authority: a run written as ``profile.dat``, or a JSON profile
    someone renamed to ``.h5``, still reads correctly.

    Falls back to :func:`profile_format` when the leading bytes match nothing
    known, so an unrecognisable file is still treated as HDF5 and produces
    h5py's own error rather than one invented here.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    file_path = Path(path)
    with file_path.open("rb") as stream:
        head = stream.read(8)
    if head.startswith(_HDF5_MAGIC):
        return FORMAT_HDF5
    if head.startswith(_GZIP_MAGIC):
        # The only gzipped shape scope-profiler writes is a JSON profile.
        return FORMAT_JSON
    stripped = head.lstrip()
    if stripped.startswith(b"{"):
        return FORMAT_JSON
    if stripped[:1] == b"<":
        return FORMAT_HTML
    return profile_format(path)


def load(path, verbose: bool = False) -> ProfilingResults:
    """Load a profile for post-processing, whatever format it is in.

    The one entry point for reading a run back::

        import scope_profiler

        results = scope_profiler.load("profiling_data.h5")
        results.print_summary()

    HDF5 and JSON profiles are both accepted, and the format is taken from
    the file's contents rather than its name (see
    :func:`sniff_profile_format`), so the caller does not have to know which
    of :func:`~scope_profiler.h5reader.read_h5` and
    :func:`~scope_profiler.json_export.read_json` it wanted. Those remain
    available for reading a file whose format is already known.

    Parameters
    ----------
    path : str | Path
        The profile to read.
    verbose : bool, optional
        Print progress while reading (default: False).

    Returns
    -------
    ProfilingResults
        The run's profiling data, with all durations in seconds.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If ``path`` is an HTML report. A report is a rendering of a run, not
        a copy of it, so there is nothing to read back.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"No profile at {str(file_path)!r}")
    kind = sniff_profile_format(file_path)
    if kind == FORMAT_HTML:
        raise ValueError(
            f"{str(file_path)!r} is an HTML report, which is write-only: it "
            "renders a run rather than storing it. Read the .h5 or .json "
            "profile it was made from.",
        )
    if kind == FORMAT_JSON:
        from scope_profiler.json_export import read_json

        return read_json(file_path, verbose=verbose)
    from scope_profiler.h5reader import read_h5

    return read_h5(file_path, verbose=verbose)


def read_profile(file_path, verbose: bool = False) -> ProfilingResults:
    """Read a profile from HDF5 or JSON, whichever the name says it is.

    Dispatches on the file name; :func:`load` dispatches on the contents and
    is the better default for a file that already exists.
    """
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
