"""CLI entry point for ``scope-profiler tags``: list the available opt-in tags.

A library registers its opt-in tags (:meth:`ProfileManager.register_opt_in_tag`)
when it is imported, so the tags of an environment are only known after its
packages are imported. This command imports them first:

- every module declared in the ``scope_profiler.opt_in_tags`` entry-point
  group, which a library adds to its ``pyproject.toml``::

      [project.entry-points."scope_profiler.opt_in_tags"]
      mylib = "mylib"

- every module named with ``--import MODULE``, for packages that do not
  declare the entry point.

It then prints each registered opt-in tag, the module that registered it, and
the regions known to carry it (see :meth:`ProfileManager.opt_in_tags`
for when a region becomes known), as text or, with ``--json``, as a JSON
document. A module that fails to import is reported as a warning.
"""

import argparse
import importlib
import json
import sys
from importlib.metadata import entry_points

from scope_profiler import profile_manager
from scope_profiler.profile_manager import ProfileManager

#: Entry-point group whose modules ``scope-profiler tags`` imports.
ENTRY_POINT_GROUP = "scope_profiler.opt_in_tags"


def _import(name, load, errors) -> None:
    """Run ``load``; a failure appends a ``{"source", "error"}`` record to ``errors``."""
    try:
        load()
    except Exception as error:
        errors.append({"source": name, "error": f"{type(error).__name__}: {error}"})


def collect_tags(imports=(), use_entry_points: bool = True) -> dict:
    """Import the tag-declaring modules and collect the opt-in tags.

    Parameters
    ----------
    imports : iterable of str, optional
        Further modules to import, after the entry points.
    use_entry_points : bool, optional
        Whether to import the modules of the ``scope_profiler.opt_in_tags``
        entry points (default: True).

    Returns
    -------
    dict
        ``{"tags": [{"tag", "module", "regions"}], "errors": [{"source",
        "error"}]}``, where ``module`` is the module that registered the tag
        and ``source`` the entry point or module that failed to import.
    """
    errors: list[dict] = []
    if use_entry_points:
        for entry_point in entry_points(group=ENTRY_POINT_GROUP):
            _import(f"entry point {entry_point.name}", entry_point.load, errors)
    for module in imports:
        _import(module, lambda module=module: importlib.import_module(module), errors)
    tags = [
        {
            "tag": tag,
            "module": profile_manager._OPT_IN_TAG_MODULES.get(tag),
            "regions": list(regions),
        }
        for tag, regions in ProfileManager.opt_in_tags().items()
    ]
    return {"tags": tags, "errors": errors}


def print_tags(collected: dict, stream=None) -> None:
    """Print the result of :func:`collect_tags` as text.

    Parameters
    ----------
    collected : dict
        The return value of :func:`collect_tags`.
    stream : file-like, optional
        Where to write (default: ``sys.stdout``).
    """
    stream = sys.stdout if stream is None else stream
    if not collected["tags"]:
        print(
            "No opt-in tags registered. Packages declare them with the "
            f"{ENTRY_POINT_GROUP!r} entry point; import others with --import MODULE.",
            file=stream,
        )
        return
    for entry in collected["tags"]:
        module = f" (registered by {entry['module']})" if entry["module"] else ""
        print(f"{entry['tag']}{module}", file=stream)
        for region in entry["regions"]:
            print(f"  {region}", file=stream)
        if not entry["regions"]:
            print("  (no regions known yet)", file=stream)
    print(
        "\nEnable a tag with `scope-profiler run --enable-tag TAG`, "
        "SCOPE_PROFILER_ENABLE_TAGS=TAG or setup(enable_tags=[TAG]); '*' enables "
        "all. Regions opened with `with sp.region(...)` are listed only once "
        "they ran.",
        file=stream,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``scope-profiler tags``."""
    parser = argparse.ArgumentParser(
        prog="scope-profiler tags",
        description=(
            "List the opt-in region tags of the installed packages and the "
            f"regions behind each. Imports the modules of the {ENTRY_POINT_GROUP!r} "
            "entry points first."
        ),
    )
    parser.add_argument(
        "--import",
        dest="imports",
        action="append",
        default=[],
        metavar="MODULE",
        help="Also import this module before listing; repeatable",
    )
    parser.add_argument(
        "--no-entry-points",
        action="store_true",
        help=f"Do not import the modules of the {ENTRY_POINT_GROUP!r} entry points",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the tags as a JSON document",
    )
    return parser


def main(argv: list | None = None) -> int:
    """Entry point for ``scope-profiler tags``.

    Returns
    -------
    int
        1 when a module named with ``--import`` failed to import, else 0.
    """
    args = build_parser().parse_args(argv)
    collected = collect_tags(args.imports, use_entry_points=not args.no_entry_points)
    for error in collected["errors"]:
        print(
            f"warning: could not import {error['source']}: {error['error']}",
            file=sys.stderr,
        )
    if args.json:
        print(json.dumps(collected, indent=2))
    else:
        print_tags(collected)
    failed = {error["source"] for error in collected["errors"]}
    return 1 if failed.intersection(args.imports) else 0
