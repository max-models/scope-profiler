"""Collection and storage of LIKWID marker results.

Counter collection happens at ``finalize()``, on top of a run that has already
completed. It is a bonus, never the point of the job, so no failure here may
cost the user their timing data. That constraint shapes the whole module:
there are three sources of counter data, tried richest first, and the risky
one is fenced off in a child process.

* **Full API, out of process** (:func:`collect_marker_results_isolated`) ---
  the richest source. After ``markerclose()`` has written the marker file,
  re-initializing the perfmon module and calling ``markerreadfile()`` exposes
  every region, for every thread, with event names, counter registers and
  LIKWID's derived metrics (``Clock [MHz]``, ``CPI``, ``Energy [J]``, ...).

  Re-initializing perfmon is also the one step that can take the interpreter
  down rather than raise: on hosts where LIKWID cannot really count (a
  virtualized CI runner with an unreadable TSC and counters disabled by
  HyperThreading, say) it has been observed to segfault. It therefore runs in
  a subprocess, where a crash costs nothing but the enrichment.

* **Marker file** (:func:`parse_marker_file`) --- the file LIKWID writes is
  plain text, so it can be read with no LIKWID calls at all and cannot fail
  catastrophically. It yields every region, thread, call count, runtime and
  raw counter value; only the *names* and the derived metrics are missing.
  This is the fallback whenever the subprocess above does not come back.

* **Marker API** (:func:`collect_region_snapshots`) --- ``markergetregion(tag)``
  read while the markers are still open. Last resort, for when the marker file
  is absent entirely; reports only the calling thread.

All three require the process to have been started under ``likwid-perfctr -m``
(or ``likwid-mpirun ... -marker``). Without that, LIKWID sets no environment
and there is nothing to collect.
"""

import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, List

import numpy as np

# Environment set by `likwid-perfctr -m` / `likwid-mpirun -marker` for each
# process it launches. LIKWID_THREADS/LIKWID_EVENTS describe the measurement,
# LIKWID_FILEPATH is where markerclose() dumps the results.
_ENV_FILEPATH = "LIKWID_FILEPATH"
_ENV_THREADS = "LIKWID_THREADS"
_ENV_EVENTS = "LIKWID_EVENTS"

#: Name of the HDF5 group holding the LIKWID results inside a (per-rank) file.
LIKWID_GROUP = "likwid"


@dataclass
class LikwidRegionResult:
    """Counter results for a single LIKWID marker region.

    The per-thread arrays are all indexed by the same thread axis, so
    ``events[e, t]`` and ``metrics[m, t]`` refer to the thread whose runtime is
    ``times[t]`` and whose CPU is ``cpus[t]``.

    Attributes
    ----------
    tag : str
        Region name as passed to ``markerstartregion``.
    group_id : int
        Index of the LIKWID event group this region was measured with.
    group_name : str
        Name of that group (e.g. ``"CLOCK"``), empty if unknown.
    cpus : list of int
        Hardware threads that took part in the region.
    times : numpy.ndarray
        Per-thread accumulated runtime in seconds, shape ``(nthreads,)``.
    call_counts : numpy.ndarray
        Per-thread number of times the region was entered, shape ``(nthreads,)``.
    event_names : list of str
        Names of the raw hardware events, length ``nevents``. **Not unique**:
        a group may program one event on several counters (see
        :attr:`event_labels`).
    counter_names : list of str
        Hardware counter register each event was programmed on (``FIXC0``,
        ``PMC1``, ``MBOX3C0``, ...), length ``nevents``. Empty for files
        written before counter names were recorded.
    events : numpy.ndarray
        Raw counter values, shape ``(nevents, nthreads)``.
    metric_names : list of str
        Names of LIKWID's derived metrics, length ``nmetrics``.
    metrics : numpy.ndarray
        Derived metric values, shape ``(nmetrics, nthreads)``.
    source : str
        Which collection path produced this result: ``"full_api"`` (perfmon
        read-back, the only one with real event names and metrics),
        ``"marker_file"`` (LIKWID's marker file parsed directly --- real
        values, placeholder event names, no metrics) or ``"marker_api"`` (a
        ``markergetregion`` snapshot of the calling thread only).
    """

    tag: str
    group_id: int = -1
    group_name: str = ""
    cpus: List[int] = field(default_factory=list)
    times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    call_counts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    event_names: List[str] = field(default_factory=list)
    counter_names: List[str] = field(default_factory=list)
    events: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    metric_names: List[str] = field(default_factory=list)
    metrics: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    source: str = "full_api"

    @property
    def event_labels(self) -> List[str]:
        """Unique per-event labels, safe to use as dict or column keys.

        ``event_names`` alone is not unique. A group such as ``MEM_DP``
        programs the same event on one counter per memory channel, so
        ``CAS_COUNT_RD`` legitimately appears eight times on a socket with
        eight channels (the ones reading zero are unpopulated). Keying
        anything by the bare name silently keeps only the last channel.

        Names that occur once are returned unchanged; repeated ones get the
        hardware counter appended --- ``CAS_COUNT_RD:MBOX0C0``,
        ``CAS_COUNT_RD:MBOX1C0``, ... --- or a positional suffix if the file
        predates counter names being recorded.
        """
        seen = Counter(self.event_names)
        labels = []
        for index, name in enumerate(self.event_names):
            if seen[name] == 1:
                labels.append(name)
            elif index < len(self.counter_names):
                labels.append(f"{name}:{self.counter_names[index]}")
            else:
                labels.append(f"{name}#{index}")
        return labels

    def as_dict(self) -> dict:
        """Return the region's results as a plain dictionary.

        Convenience for callers that want to build a DataFrame or dump the
        numbers without touching the array layout.
        """
        return {
            "tag": self.tag,
            "group_id": self.group_id,
            "group_name": self.group_name,
            "cpus": list(self.cpus),
            "times": self.times,
            "call_counts": self.call_counts,
            "event_names": list(self.event_names),
            "counter_names": list(self.counter_names),
            "event_labels": self.event_labels,
            "events": self.events,
            "metric_names": list(self.metric_names),
            "metrics": self.metrics,
            "source": self.source,
        }


def likwid_environment() -> dict:
    """Return the LIKWID environment of the current process.

    Returns
    -------
    dict
        The ``LIKWID_*`` variables set by the launcher. Empty when the process
        was not started under ``likwid-perfctr -m``.
    """
    return {k: v for k, v in os.environ.items() if k.startswith("LIKWID_")}


def markers_available() -> bool:
    """Whether this process runs under the LIKWID marker API.

    ``markerinit()`` silently degrades to a no-op when LIKWID's environment is
    absent, so this is what distinguishes "there will be counter data" from
    "the script was started as a plain ``python script.py``".
    """
    return bool(os.environ.get(_ENV_FILEPATH) and os.environ.get(_ENV_EVENTS))


def _parse_cpus(value: str) -> List[int]:
    """Parse LIKWID's comma-separated hardware-thread list."""
    cpus = []
    for item in value.split(","):
        item = item.strip()
        if item:
            try:
                cpus.append(int(item))
            except ValueError:
                continue
    return cpus


def collect_region_snapshots(pylikwid, region_names: Iterable[str]) -> List[dict]:
    """Read the current counter values for each named region.

    Uses the marker API's ``markergetregion``, so it must be called *before*
    ``markerclose()`` and reports only the calling thread. Regions that LIKWID
    does not know about (never entered, or measured under a different name) are
    skipped.

    Parameters
    ----------
    pylikwid : module
        The imported ``pylikwid`` module.
    region_names : iterable of str
        Region tags to query.

    Returns
    -------
    list of dict
        One entry per region that returned data, with keys ``tag``,
        ``nevents``, ``events``, ``time`` and ``count``.
    """
    snapshots = []
    for tag in region_names:
        try:
            nevents, values, time, count = pylikwid.markergetregion(tag)
        except Exception:
            # An unknown tag raises rather than returning empty results; a
            # region the user never entered is not an error worth failing on.
            continue
        if not nevents or count <= 0:
            continue
        snapshots.append(
            {
                "tag": tag,
                "nevents": int(nevents),
                "events": np.asarray(values, dtype=np.float64),
                "time": float(time),
                "count": int(count),
            }
        )
    return snapshots


def snapshots_to_results(snapshots: Iterable[dict]) -> List[LikwidRegionResult]:
    """Convert marker-API snapshots into the common result structure.

    Used as the fallback when the full API cannot re-open the counters, so the
    snapshots taken before ``markerclose()`` can still be written to HDF5.
    """
    results = []
    for snap in snapshots:
        values = snap["events"].reshape(-1, 1)
        results.append(
            LikwidRegionResult(
                tag=snap["tag"],
                cpus=[],
                times=np.array([snap["time"]], dtype=np.float64),
                call_counts=np.array([snap["count"]], dtype=np.int64),
                # The marker API hands out bare numbers; without a perfmon
                # group there is no way to name them.
                event_names=[f"event_{i}" for i in range(len(values))],
                events=values,
                source="marker_api",
            )
        )
    return results


def parse_marker_file(path=None) -> List[LikwidRegionResult]:
    """Read LIKWID's marker file directly, without calling into LIKWID.

    The file ``markerclose()`` writes is plain text::

        <nthreads> <nregions> <ngroups>
        <region_id>:<tag>-<group_id>          (one line per region)
        <region_id> <group_id> <cpu> <call_count> <time> <nevents> <values...>

    Parsing it here is the crash-proof path: it touches no counters and calls
    no LIKWID function, so it cannot take the interpreter down the way
    re-initializing perfmon can. The cost is that event names, counter
    registers and derived metrics are not in the file --- those exist only
    inside LIKWID's group definitions --- so events come back positionally
    named and the metric list is empty.

    Parameters
    ----------
    path : str, optional
        Marker file to read (default: ``$LIKWID_FILEPATH``).

    Returns
    -------
    list of LikwidRegionResult
        One entry per region, ordered by region id. Empty if the file is
        missing or malformed.
    """
    path = path or os.environ.get(_ENV_FILEPATH)
    if not path or not os.path.exists(path):
        return []

    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            lines = [line.strip() for line in handle if line.strip()]
    except OSError:
        return []

    if not lines:
        return []

    try:
        num_regions = int(lines[0].split()[1])

        tags, groups = {}, {}
        for line in lines[1 : 1 + num_regions]:
            region_id, _, rest = line.partition(":")
            # The group id is appended to the tag as "-<gid>"; a tag may itself
            # contain a dash, so split from the right.
            tag, _, group_id = rest.rpartition("-")
            tags[int(region_id)] = tag
            groups[int(region_id)] = int(group_id)

        per_region = {}
        for line in lines[1 + num_regions :]:
            fields = line.split()
            if len(fields) < 6:
                continue
            region_id = int(fields[0])
            cpu = int(fields[2])
            count = int(float(fields[3]))
            time = float(fields[4])
            num_events = int(fields[5])
            values = [float(v) for v in fields[6 : 6 + num_events]]
            per_region.setdefault(region_id, []).append((cpu, count, time, values))
    except (ValueError, IndexError):
        # A truncated or unexpected file is not worth failing finalize() over.
        return []

    results = []
    for region_id in sorted(per_region):
        threads = per_region[region_id]
        num_events = max((len(values) for *_, values in threads), default=0)
        events = np.zeros((num_events, len(threads)), dtype=np.float64)
        for index, (_, _, _, values) in enumerate(threads):
            events[: len(values), index] = values
        results.append(
            LikwidRegionResult(
                tag=tags.get(region_id, str(region_id)),
                group_id=groups.get(region_id, -1),
                cpus=[cpu for cpu, _, _, _ in threads],
                times=np.array([t for _, _, t, _ in threads], dtype=np.float64),
                call_counts=np.array([c for _, c, _, _ in threads], dtype=np.int64),
                event_names=[f"event_{i}" for i in range(num_events)],
                events=events,
                source="marker_file",
            )
        )
    return results


def collect_marker_results(pylikwid) -> List[LikwidRegionResult]:
    """Read every region of the run back from LIKWID's marker file.

    Must be called *after* ``markerclose()``, which is what writes the file.
    Because ``markerclose()`` also tears the perfmon module down, the event
    sets named in ``LIKWID_EVENTS`` are re-registered first; that is what makes
    event and metric names available for the values in the file.

    Parameters
    ----------
    pylikwid : module
        The imported ``pylikwid`` module.

    Returns
    -------
    list of LikwidRegionResult
        One entry per region recorded by LIKWID, in file order. Empty when the
        process is not running under the marker API, or when the performance
        counters cannot be re-opened.
    """
    filepath = os.environ.get(_ENV_FILEPATH)
    if not filepath or not os.path.exists(filepath):
        return []

    cpus = _parse_cpus(os.environ.get(_ENV_THREADS, ""))
    event_string = os.environ.get(_ENV_EVENTS, "")
    if not cpus or not event_string:
        return []

    try:
        if pylikwid.init(cpus) != 0:
            return []
    except Exception:
        return []

    try:
        # LIKWID separates the event sets of a multi-group run with "|", and
        # numbers the groups in that order -- the same ids markerregiongroup()
        # reports, so re-adding them in order restores the mapping.
        group_names = {}
        for event_set in event_string.split("|"):
            event_set = event_set.strip()
            if not event_set:
                continue
            gid = pylikwid.addeventset(event_set)
            if gid < 0:
                continue
            group_names[gid] = pylikwid.getnameofgroup(gid) or event_set

        pylikwid.markerreadfile(filepath)
        num_regions = pylikwid.markernumregions()
        if not isinstance(num_regions, int) or num_regions <= 0:
            # Negative values are LIKWID's error codes, not region counts.
            return []

        results = []
        for r in range(num_regions):
            gid = pylikwid.markerregiongroup(r)
            nthreads = pylikwid.markerregionthreads(r)
            nevents = pylikwid.markerregionevents(r)
            if nthreads <= 0:
                continue

            try:
                nmetrics = pylikwid.getnumberofmetrics(gid)
            except Exception:
                nmetrics = 0

            events = np.zeros((nevents, nthreads), dtype=np.float64)
            metrics = np.zeros((nmetrics, nthreads), dtype=np.float64)
            times = np.zeros(nthreads, dtype=np.float64)
            counts = np.zeros(nthreads, dtype=np.int64)

            for t in range(nthreads):
                times[t] = pylikwid.markerregiontime(r, t)
                counts[t] = pylikwid.markerregioncount(r, t)
                for e in range(nevents):
                    events[e, t] = pylikwid.markerregionresult(r, e, t)
                for m in range(nmetrics):
                    metrics[m, t] = pylikwid.markerregionmetric(r, m, t)

            results.append(
                LikwidRegionResult(
                    tag=pylikwid.markerregiontag(r),
                    group_id=int(gid),
                    group_name=group_names.get(gid, ""),
                    cpus=[int(c) for c in (pylikwid.markerregioncpulist(r) or [])],
                    times=times,
                    call_counts=counts,
                    event_names=[
                        pylikwid.getnameofevent(gid, e) for e in range(nevents)
                    ],
                    # The register each event sits on. This is what tells the
                    # eight identically-named CAS_COUNT_RD entries of a group
                    # like MEM_DP apart (one per memory channel).
                    counter_names=[
                        pylikwid.getnameofcounter(gid, e) for e in range(nevents)
                    ],
                    events=events,
                    metric_names=[
                        pylikwid.getnameofmetric(gid, m) for m in range(nmetrics)
                    ],
                    metrics=metrics,
                    source="full_api",
                )
            )
        return results
    except Exception:
        # Counter collection is a bonus on top of the timing data: a LIKWID
        # failure here must not take the whole finalize() down with it.
        return []
    finally:
        try:
            pylikwid.finalize()
        except Exception:
            pass


def _result_to_json(result: LikwidRegionResult) -> dict:
    """Serialize a result so it can cross a process boundary."""
    return {
        "tag": result.tag,
        "group_id": int(result.group_id),
        "group_name": result.group_name,
        "cpus": [int(c) for c in result.cpus],
        "times": [float(t) for t in result.times],
        "call_counts": [int(c) for c in result.call_counts],
        "event_names": list(result.event_names),
        "counter_names": list(result.counter_names),
        "events": [[float(v) for v in row] for row in result.events],
        "metric_names": list(result.metric_names),
        "metrics": [[float(v) for v in row] for row in result.metrics],
        "source": result.source,
    }


def _result_from_json(payload: dict) -> LikwidRegionResult:
    """Rebuild a result from :func:`_result_to_json`."""
    num_threads = len(payload["times"])

    def matrix(rows):
        if not rows:
            return np.zeros((0, num_threads), dtype=np.float64)
        return np.array(rows, dtype=np.float64)

    return LikwidRegionResult(
        tag=payload["tag"],
        group_id=payload["group_id"],
        group_name=payload["group_name"],
        cpus=payload["cpus"],
        times=np.array(payload["times"], dtype=np.float64),
        call_counts=np.array(payload["call_counts"], dtype=np.int64),
        event_names=payload["event_names"],
        counter_names=payload.get("counter_names", []),
        events=matrix(payload["events"]),
        metric_names=payload["metric_names"],
        metrics=matrix(payload["metrics"]),
        source=payload.get("source", "full_api"),
    )


def collect_marker_results_isolated(timeout: float = 120.0):
    """Run the perfmon read-back in a child process and return its results.

    :func:`collect_marker_results` has to re-initialize LIKWID's perfmon
    module, and on hosts where the counters are not really usable that can
    abort the process outright instead of raising --- which at ``finalize()``
    time would destroy a completed run's output. Running it behind a process
    boundary turns that worst case into a missing enrichment.

    Parameters
    ----------
    timeout : float, optional
        Seconds to wait for the child before giving up (default: 120).

    Returns
    -------
    list of LikwidRegionResult or None
        ``None`` when the child could not deliver results (crashed, timed out,
        or LIKWID refused to re-open the counters), which tells the caller to
        fall back to :func:`parse_marker_file`.
    """
    if not markers_available():
        return None

    # The child writes to a file rather than stdout: LIKWID itself prints
    # warnings and error banners on both streams, which would corrupt JSON.
    handle, out_path = tempfile.mkstemp(prefix="likwid_results_", suffix=".json")
    os.close(handle)

    try:
        env = dict(os.environ)
        # The child must import the same scope_profiler (and find pylikwid)
        # as this process, however this process was started.
        env["PYTHONPATH"] = os.pathsep.join(
            [path for path in sys.path if path] + [env.get("PYTHONPATH", "")]
        ).strip(os.pathsep)

        subprocess.run(
            [sys.executable, "-m", "scope_profiler.likwid_data", out_path],
            capture_output=True,
            env=env,
            timeout=timeout,
            check=False,
        )
        # Deliberately not gated on the return code. LIKWID has been seen to
        # abort during interpreter teardown, i.e. after the results were
        # written; a complete JSON document is proof enough that the work
        # finished, and a crash mid-write leaves one that will not parse.
        with open(out_path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass

    if not payload:
        return None
    return [_result_from_json(item) for item in payload]


def _subprocess_main(argv=None) -> int:
    """Entry point of the isolated collector (``python -m ...likwid_data``).

    Not part of the public API: this is the body that runs behind the process
    boundary set up by :func:`collect_marker_results_isolated`.
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv:
        print(
            "usage: python -m scope_profiler.likwid_data <output.json>", file=sys.stderr
        )
        return 2

    from scope_profiler.profile_config import _import_pylikwid

    results = collect_marker_results(_import_pylikwid())
    with open(argv[0], "w", encoding="utf-8") as fh:
        json.dump([_result_to_json(result) for result in results], fh)
    return 0


def _h5_safe(tag: str) -> str:
    """Escape a region tag for use as an HDF5 group name.

    ``/`` is the group separator in HDF5, so a tag containing one would silently
    create nested groups. The original tag is always kept in the ``tag``
    attribute, which is what readers should use.
    """
    return tag.replace("/", "|")


def write_likwid_results(
    h5file,
    results: Iterable[LikwidRegionResult],
    environment: dict | None = None,
) -> None:
    """Write collected LIKWID results into an open HDF5 file.

    Creates (replacing any previous copy) a ``likwid`` group holding one
    subgroup per region under ``likwid/regions/<tag>``.

    Parameters
    ----------
    h5file : h5py.File or h5py.Group
        Destination, typically the per-rank profiling file.
    results : iterable of LikwidRegionResult
        Regions to store.
    environment : dict, optional
        LIKWID environment variables to record as attributes on the group, so
        the event set a file was measured with stays with the data.
    """
    import h5py

    results = list(results)

    if LIKWID_GROUP in h5file:
        del h5file[LIKWID_GROUP]
    grp = h5file.create_group(LIKWID_GROUP)

    for key, value in (environment or {}).items():
        grp.attrs[key] = value
    grp.attrs["num_regions"] = len(results)

    regions_grp = grp.create_group("regions")
    for result in results:
        rgrp = regions_grp.create_group(_h5_safe(result.tag))
        rgrp.attrs["tag"] = result.tag
        rgrp.attrs["group_id"] = result.group_id
        rgrp.attrs["group_name"] = result.group_name
        rgrp.attrs["source"] = result.source
        # Explicit string dtype: h5py would otherwise store these as
        # fixed-width bytes, and an empty list has no inferable dtype at all.
        rgrp.attrs.create(
            "event_names", list(result.event_names), dtype=h5py.string_dtype()
        )
        rgrp.attrs.create(
            "counter_names", list(result.counter_names), dtype=h5py.string_dtype()
        )
        rgrp.attrs.create(
            "metric_names", list(result.metric_names), dtype=h5py.string_dtype()
        )
        rgrp.create_dataset("cpus", data=np.asarray(result.cpus, dtype=np.int64))
        rgrp.create_dataset("times", data=result.times)
        rgrp.create_dataset("call_counts", data=result.call_counts)
        rgrp.create_dataset("events", data=result.events)
        rgrp.create_dataset("metrics", data=result.metrics)


if __name__ == "__main__":  # pragma: no cover - exercised via a subprocess
    raise SystemExit(_subprocess_main())
