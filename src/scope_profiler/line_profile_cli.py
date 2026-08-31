"""CLI for displaying line-profiler records persisted in HDF5 files."""

import argparse
import linecache
import re
import sys
from html import escape
from io import StringIO

from tabulate import tabulate

from scope_profiler.h5reader import read_h5


def _source_lines(record):
    """Return source lines with the function's common indentation removed.

    ``tabulate`` strips leading whitespace from cells, so an invisible
    sentinel is used on lines that retain indentation.  This mirrors the TUI
    rendering and keeps nested Python blocks readable in the CLI table.
    """
    linecache.checkcache(record["filename"])
    sources = [
        linecache.getline(record["filename"], int(line))
        .rstrip("\r\n")
        .expandtabs(4)
        or "<source unavailable>"
        for line in record["line_numbers"]
    ]
    base_indentation = (
        len(sources[0]) - len(sources[0].lstrip(" ")) if sources else 0
    )
    formatted = []
    for source in sources:
        indentation = len(source) - len(source.lstrip(" "))
        dedent = min(base_indentation, indentation)
        source = source[dedent:]
        if indentation - dedent:
            # tabulate strips whitespace at the start of a cell.  The
            # zero-width prefix preserves the real indentation on display.
            source = "\N{ZERO WIDTH SPACE}" + source
        formatted.append(source)
    return formatted


def print_line_profile(
    file_path,
    ranks=None,
    function=None,
    region=None,
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
    results = read_h5(file_path)
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
            record_count += 1
            unit = record["unit"]
            print(
                f"\nRank {rank} | {record['region']} | {record['function']} "
                f"({record['filename']}:{record['first_lineno']})",
                file=output_stream,
            )
            total_time = float(record["times"].sum())
            table_rows = []
            sources = _source_lines(record)
            for (line, hits, elapsed), source in zip(
                zip(record["line_numbers"], record["hits"], record["times"]),
                sources,
            ):
                seconds = float(elapsed) * unit
                per_hit = seconds / int(hits) if hits else 0.0
                percent = float(elapsed) / total_time * 100 if total_time else 0.0
                table_rows.append(
                    [
                        int(line),
                        int(hits),
                        f"{seconds:.6g}",
                        f"{per_hit:.6g}",
                        f"{percent:.2f}",
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
        "--rank",
        type=int,
        action="append",
        dest="ranks",
        help="rank to show (repeatable)",
    )
    parser.add_argument(
        "--function", help="regular expression selecting function names"
    )
    parser.add_argument("--region", help="regular expression selecting region names")
    args = parser.parse_args(argv)
    print_line_profile(
        args.file, ranks=args.ranks, function=args.function, region=args.region
    )
    return 0
