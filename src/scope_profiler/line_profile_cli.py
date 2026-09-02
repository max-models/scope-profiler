"""CLI for displaying line-profiler records persisted in HDF5 files."""

import argparse
import linecache
import re
import sys
from html import escape
from io import StringIO

from tabulate import tabulate

from scope_profiler.post_processing import parse_ranks
from scope_profiler.profile_io import read_profile
from scope_profiler.summary import _name_selected


def _line_profile_rows(record):
    """Return timed lines plus source-only context rows.

    ``tabulate`` strips leading whitespace from cells, so an invisible
    sentinel is used on lines that retain indentation.  Source-only rows have
    no timing values; they make unexecuted branches visible and preserve the
    relative indentation used by the TUI.
    """
    linecache.checkcache(record["filename"])
    timed_lines = [int(line) for line in record["line_numbers"]]
    sources = {
        line: linecache.getline(record["filename"], line).rstrip("\r\n").expandtabs(4)
        for line in range(min(timed_lines, default=0), max(timed_lines, default=-1) + 1)
    }
    if not any(sources.values()):
        sources = {line: "<source unavailable>" for line in timed_lines}

    base_source = sources.get(timed_lines[0], "") if timed_lines else ""
    base_indentation = (
        len(base_source) - len(base_source.lstrip(" ")) if base_source else 0
    )
    timings = dict(zip(timed_lines, zip(record["hits"], record["times"])))
    rows = []
    for line, source in sources.items():
        indentation = len(source) - len(source.lstrip(" "))
        dedent = min(base_indentation, indentation)
        source = source[dedent:]
        indentation -= dedent
        if indentation:
            # tabulate strips whitespace at the start of a cell.  The
            # zero-width prefix preserves the real indentation on display.
            source = "\N{ZERO WIDTH SPACE}" + source
        timing = timings.get(line)
        rows.append((line, *(timing or (None, None)), source))
    return rows


def print_line_profile(
    file_path,
    ranks=None,
    function=None,
    region=None,
    include=None,
    exclude=None,
    stream=None,
    display_html=False,
):
    """Print persisted line-profiler timings from one HDF5 profile.

    Set ``display_html=True`` when calling from a Jupyter notebook to render
    the complete output in a whitespace-preserving ``<pre>`` block.  The
    default remains plain text for the command-line interface.
    """
    output_stream = StringIO() if display_html else stream
    output_stream = sys.stdout if output_stream is None else output_stream
    results = read_profile(file_path)
    selected_ranks = results.line_profile.keys() if ranks is None else ranks
    function_re = re.compile(function) if function else None
    region_re = re.compile(region) if region else None
    record_count = 0

    print(f"Line profile: {results.file_path}", file=output_stream)
    for rank in sorted(selected_ranks):
        for record in results.line_profile.get(rank, []):
            if function_re and not function_re.search(record["function"]):
                continue
            if region_re and not region_re.search(record["region"]):
                continue
            if not _name_selected(record["region"], include, exclude):
                continue
            record_count += 1
            unit = record["unit"]
            print(
                f"\nRank {rank} | {record['region']} | {record['function']} "
                f"({record['filename']}:{record['first_lineno']})",
                file=output_stream,
            )
            total_time = float(record["times"].sum())
            table_rows = []
            for line, hits, elapsed, source in _line_profile_rows(record):
                seconds = float(elapsed) * unit if elapsed is not None else None
                per_hit = seconds / int(hits) if hits and seconds is not None else None
                percent = (
                    float(elapsed) / total_time * 100
                    if elapsed is not None and total_time
                    else None
                )
                table_rows.append(
                    [
                        int(line),
                        int(hits) if hits is not None else "",
                        f"{seconds:.6g}" if seconds is not None else "",
                        f"{per_hit:.6g}" if per_hit is not None else "",
                        f"{percent:.2f}" if percent is not None else "",
                        source,
                    ]
                )
            for table_line in tabulate(
                table_rows,
                headers=("line", "hits", "time [s]", "per hit [s]", "% time", "source"),
                tablefmt="rounded_outline",
                disable_numparse=True,
            ).splitlines():
                print(table_line, file=output_stream)

    if record_count == 0:
        print("\nNo line-profile records matched.", file=output_stream)

    if display_html:
        from IPython.display import HTML, display

        display(HTML(f"<pre>{escape(output_stream.getvalue())}</pre>"))


def main(argv=None):
    """Handle ``scope-profiler line-profile``."""
    parser = argparse.ArgumentParser(
        prog="scope-profiler line-profile",
        description="Print persisted line-profiler timings from an HDF5 profile.",
    )
    parser.add_argument("file", help="HDF5 profiling file")
    parser.add_argument(
        "--ranks",
        "--rank",
        "-r",
        nargs="*",
        dest="ranks",
        default=["0"],
        metavar="RANK",
        help="Ranks to include. Supports comma-separated values and ranges (default: 0).",
    )
    parser.add_argument(
        "--include",
        "-i",
        nargs="*",
        help="Region names to include (regex patterns).",
    )
    parser.add_argument(
        "--exclude",
        "-e",
        nargs="*",
        help="Region names to exclude (regex patterns).",
    )
    parser.add_argument(
        "--function", help="regular expression selecting function names"
    )
    parser.add_argument("--region", help="regular expression selecting region names")
    args = parser.parse_args(argv)
    ranks = None
    if args.ranks:
        ranks = sorted({rank for spec in args.ranks for rank in parse_ranks(spec)})
    print_line_profile(
        args.file,
        ranks=ranks,
        function=args.function,
        region=args.region,
        include=args.include,
        exclude=args.exclude,
    )
    return 0
