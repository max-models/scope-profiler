"""Collection and storage of LIKWID marker results.

LIKWID's marker API only accumulates counters while the process runs; the
numbers themselves are handed out in two different ways, and this module uses
both:

* **Marker API** (:func:`collect_region_snapshots`) --- ``markergetregion(tag)``
  returns the raw counter values for a single region on the calling thread.
  It works at any point while the markers are open, needs no extra privileges,
  but gives neither event names nor derived metrics.
* **Full API** (:func:`collect_marker_results`) --- after ``markerclose()`` has
  written the marker file, re-initializing the perfmon module and calling
  ``markerreadfile()`` exposes *every* region of the run, for every thread,
  with event names, per-thread call counts and LIKWID's derived metrics
  (``Clock [MHz]``, ``CPI``, ``Energy [J]``, ...).

The full API is the richer of the two but needs to re-open the performance
counters, which can fail (no access daemon, no permissions, no marker file).
The snapshots are therefore taken first, while the markers are still open, and
are used as a fallback so a run always ends up with *some* counter data in the
HDF5 file.

Both paths require the process to have been started under ``likwid-perfctr -m``
(or ``likwid-mpirun ... -marker``). Without that, LIKWID sets no environment
and there is nothing to collect.
"""

import os
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
        Names of the raw hardware events, length ``nevents``.
    events : numpy.ndarray
        Raw counter values, shape ``(nevents, nthreads)``.
    metric_names : list of str
        Names of LIKWID's derived metrics, length ``nmetrics``.
    metrics : numpy.ndarray
        Derived metric values, shape ``(nmetrics, nthreads)``.
    source : str
        ``"full_api"`` when read back from the marker file, ``"marker_api"``
        when taken from a ``markergetregion`` snapshot (no metrics, and event
        names are placeholders).
    """

    tag: str
    group_id: int = -1
    group_name: str = ""
    cpus: List[int] = field(default_factory=list)
    times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    call_counts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    event_names: List[str] = field(default_factory=list)
    events: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    metric_names: List[str] = field(default_factory=list)
    metrics: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    source: str = "full_api"

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
            "metric_names", list(result.metric_names), dtype=h5py.string_dtype()
        )
        rgrp.create_dataset("cpus", data=np.asarray(result.cpus, dtype=np.int64))
        rgrp.create_dataset("times", data=result.times)
        rgrp.create_dataset("call_counts", data=result.call_counts)
        rgrp.create_dataset("events", data=result.events)
        rgrp.create_dataset("metrics", data=result.metrics)
