"""Python calling Fortran, profiled as one run.

The interesting case for this library is a Python driver over Fortran kernels:
both sides mark regions, in one process, and the user wants a single profile.
These tests build a real f2py extension around the shipped Fortran module and
check that the two halves land in one file, correctly nested.

They skip unless a Fortran compiler and f2py's build backend (meson + ninja)
are all available.
"""

import importlib.util
import shutil
import subprocess
import sys

import pytest

from scope_profiler import ProfileManager, read_h5
from scope_profiler.native_trace import fortran_source_path

from .test_fortran_api import COMPILER

MODULE_SOURCE = fortran_source_path()

HAVE_F2PY_BACKEND = all(
    importlib.util.find_spec(name) is not None for name in ("mesonbuild", "ninja")
) or all(shutil.which(name) is not None for name in ("meson", "ninja"))

pytestmark = [
    pytest.mark.skipif(COMPILER is None, reason="no Fortran compiler on PATH"),
    pytest.mark.skipif(
        not MODULE_SOURCE.exists(), reason="Fortran sources not shipped"
    ),
    pytest.mark.skipif(not HAVE_F2PY_BACKEND, reason="f2py needs meson and ninja"),
]

#: A Fortran kernel library that profiles its own internals, the way a real
#: one would: the driver never sees these regions, they come from inside.
KERNELS = """
module kernels
   use scope_profiler
   implicit none
contains

   subroutine start_profiling(prefix, rank)
      character(len=*), intent(in) :: prefix
      integer, intent(in) :: rank
      call sp_init(prefix, rank=rank)
   end subroutine start_profiling

   subroutine stop_profiling()
      call sp_finalize()
   end subroutine stop_profiling

   subroutine solve(n, result)
      integer, intent(in) :: n
      real(kind=8), intent(out) :: result
      integer :: i, assemble_id, factor_id

      assemble_id = sp_region("fortran:assemble")
      factor_id = sp_region("fortran:factorize")

      call sp_begin(assemble_id)
      result = 0.0d0
      do i = 1, n
         result = result + sqrt(real(i, kind=8))
      end do
      call sp_end(assemble_id)

      call sp_begin(factor_id)
      do i = 1, n/2
         result = result + log(real(i, kind=8) + 1.0d0)
      end do
      call sp_end(factor_id)
   end subroutine solve

end module kernels
"""


@pytest.fixture(scope="module")
def kernels(tmp_path_factory):
    """Build the f2py extension once and import it."""
    build_dir = tmp_path_factory.mktemp("f2py")
    (build_dir / "kernels.f90").write_text(KERNELS)
    shutil.copy(MODULE_SOURCE, build_dir / "scope_profiler.f90")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "numpy.f2py",
            "-c",
            "scope_profiler.f90",
            "kernels.f90",
            "-m",
            "kernels",
            "--quiet",
            "--backend",
            "meson",
        ],
        cwd=build_dir,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode != 0:
        pytest.skip(f"f2py build failed:\n{result.stdout}\n{result.stderr}")

    sys.path.insert(0, str(build_dir))
    try:
        import kernels as extension
    except ImportError as exc:  # pragma: no cover - build succeeded but import failed
        pytest.skip(f"could not import the f2py extension: {exc}")
    finally:
        sys.path.remove(str(build_dir))

    yield extension.kernels, build_dir


@pytest.fixture(autouse=True)
def reset_manager():
    yield
    ProfileManager._reset()


def test_one_file_holds_both_languages(kernels, tmp_path):
    """The headline: a mixed run produces a single, ordinary profile."""
    extension, _ = kernels
    output = tmp_path / "mixed.h5"

    ProfileManager.setup(file_path=str(output))
    extension.start_profiling(str(tmp_path / "trace"), 0)

    for _ in range(3):
        with ProfileManager.profile_region("python:step"):
            with ProfileManager.profile_region("python:call_fortran"):
                extension.solve(100000)

    extension.stop_profiling()
    ProfileManager.finalize(verbose=False, native_traces=tmp_path)

    results = read_h5(output)

    assert sorted(results.region_names) == [
        "fortran:assemble",
        "fortran:factorize",
        "python:call_fortran",
        "python:step",
    ]
    assert results["python:step"].num_calls == 3
    assert results["fortran:assemble"].num_calls == 3


def test_the_two_languages_nest_correctly(kernels, tmp_path):
    """Fortran regions must sit inside the Python region that called them.

    This is what the shared clock buys: without it the two sets of timestamps
    would be incomparable and the call stack meaningless.
    """
    extension, _ = kernels

    ProfileManager.setup(deactivate_file_output=True)
    extension.start_profiling(str(tmp_path / "trace"), 0)

    with ProfileManager.profile_region("python:call_fortran"):
        extension.solve(100000)

    extension.stop_profiling()
    results = ProfileManager.finalize(
        verbose=False, return_results=True, native_traces=tmp_path
    )

    call = results["python:call_fortran"][0]
    start, end = int(call.start_times_ns[0]), int(call.end_times_ns[0])

    for name in ("fortran:assemble", "fortran:factorize"):
        region = results[name][0]
        assert int(region.start_times_ns.min()) >= start, f"{name} started too early"
        assert int(region.end_times_ns.max()) <= end, f"{name} ended too late"

    # The reconstructed call stack sees one tree, not two.
    stack = results.call_stack(rank=0)
    names = [call["name"] for call in stack]
    assert names[0] == "python:call_fortran"
    assert {"fortran:assemble", "fortran:factorize"} <= set(names[1:])
    assert all(entry["depth"] > 0 for entry in stack[1:])


def test_a_name_used_by_both_sides_is_refused(kernels, tmp_path):
    """Merging them would double-count the wrapper and the wrapped."""
    extension, _ = kernels

    ProfileManager.setup(deactivate_file_output=True)
    extension.start_profiling(str(tmp_path / "trace"), 0)
    with ProfileManager.profile_region("fortran:assemble"):  # same name on purpose
        extension.solve(1000)
    extension.stop_profiling()

    with pytest.raises(ValueError, match="recorded by both"):
        ProfileManager.finalize(
            verbose=False, return_results=True, native_traces=tmp_path
        )


def test_only_this_ranks_trace_is_folded_in(kernels, tmp_path):
    """Each rank picks up its own trace, so the MPI merge needs no special case."""
    extension, _ = kernels

    ProfileManager.setup(deactivate_file_output=True)
    # A trace belonging to a different rank must be ignored by rank 0.
    extension.start_profiling(str(tmp_path / "other"), 7)
    extension.solve(1000)
    extension.stop_profiling()

    with ProfileManager.profile_region("python:only"):
        pass

    results = ProfileManager.finalize(
        verbose=False, return_results=True, native_traces=tmp_path
    )

    assert (tmp_path / "other_rank00007.spt").exists()
    assert results.region_names == ["python:only"]


def test_merging_after_the_fact(kernels, tmp_path):
    """The offline route: two separate runs, combined later."""
    from scope_profiler.native_trace import load_traces
    from scope_profiler.results import merge_results

    extension, _ = kernels

    ProfileManager.setup(deactivate_file_output=True)
    extension.start_profiling(str(tmp_path / "trace"), 0)
    with ProfileManager.profile_region("python:driver"):
        extension.solve(50000)
    extension.stop_profiling()
    python_results = ProfileManager.finalize(verbose=False, return_results=True)

    combined = merge_results(
        python_results, load_traces(tmp_path), label="mixed", file_path="combined.h5"
    )

    assert sorted(combined.region_names) == [
        "fortran:assemble",
        "fortran:factorize",
        "python:driver",
    ]
    assert combined.label == "mixed"
    # And it behaves like any other result set.
    assert combined.to_dataframe().shape[0] == 3
