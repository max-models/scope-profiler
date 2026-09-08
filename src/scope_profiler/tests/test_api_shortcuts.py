"""The module-level API, the ``region`` name, option groups, and ``load``.

These cover the four surfaces added on top of the original
``ProfileManager``-only API: the module-level shortcuts, ``region()`` and its
``profile_region`` alias, the grouped option bags, and format-sniffing
``load()``.
"""

import gzip
import json

import pytest

from scope_profiler import (
    GPUOptions,
    HDF5Options,
    MemrayOptions,
    ProfileManager,
    ProfilingOptions,
)
from scope_profiler.profile_config import (
    _CONFIG_FIELDS,
    SetupOptions,
    load_profiling_config,
)
from scope_profiler.profile_io import (
    FORMAT_HDF5,
    FORMAT_HTML,
    FORMAT_JSON,
    sniff_profile_format,
)


@pytest.fixture(autouse=True)
def _reset():
    yield
    ProfileManager._reset()


# --- module-level shortcuts -------------------------------------------------


def test_module_level_names_are_the_manager_methods():
    import scope_profiler as sp

    assert sp.setup.__func__ is ProfileManager.setup.__func__
    assert sp.finalize.__func__ is ProfileManager.finalize.__func__
    assert sp.region.__func__ is ProfileManager.region.__func__
    assert sp.session.__func__ is ProfileManager.session.__func__
    assert sp.profile.__func__ is ProfileManager.profile.__func__


def test_a_whole_run_through_the_module_level_api(tmp_path):
    import scope_profiler as sp

    with sp.session(
        file_path=str(tmp_path / "run.h5"),
        verbose=False,
        return_results=True,
    ) as run:

        @sp.profile
        def decorated():
            sum(range(100))

        with sp.region("solve"):
            decorated()

    assert "solve" in run.results.region_names
    assert "decorated" in run.results.region_names


# --- region / profile_region ------------------------------------------------


def test_region_and_profile_region_are_the_same_method():
    assert ProfileManager.region.__func__ is ProfileManager.profile_region.__func__


def test_both_spellings_reach_the_same_region():
    ProfileManager.setup(deactivate_file_output=True)

    assert ProfileManager.region("shared") is ProfileManager.profile_region("shared")


# --- option groups ----------------------------------------------------------


def test_groups_expand_to_the_flat_setting_names():
    options = ProfilingOptions(
        memray=MemrayOptions(enabled=True, native_traces=True),
        gpu=GPUOptions(timing=True, backend="cupy", nvtx=True),
        hdf5=HDF5Options(compression="gzip", compression_level=4, chunk_size=8),
    )

    assert options.to_kwargs() == {
        "use_memray": True,
        "memray_native_traces": True,
        "use_gpu_timing": True,
        "gpu_timing_backend": "cupy",
        "use_nvtx": True,
        "hdf5_compression": "gzip",
        "hdf5_compression_level": 4,
        "hdf5_chunk_size": 8,
    }


def test_group_settings_reach_the_config(tmp_path):
    options = ProfilingOptions(
        file_path=str(tmp_path / "run.h5"),
        hdf5=HDF5Options(compression="gzip", compression_level=4),
    )

    ProfileManager.setup(options=options)
    config = ProfileManager.get_config()

    assert config.hdf5_compression == "gzip"
    assert config.hdf5_compression_level == 4


def test_setting_one_thing_both_ways_is_an_error():
    options = ProfilingOptions(
        hdf5_compression="lzf",
        hdf5=HDF5Options(compression="gzip"),
    )

    with pytest.raises(ValueError, match="both directly and on the 'hdf5'"):
        options.to_kwargs()


def test_an_empty_group_contributes_nothing():
    assert ProfilingOptions(hdf5=HDF5Options()).to_kwargs() == {}


# --- TOML group tables ------------------------------------------------------


def _write_toml(tmp_path, text):
    path = tmp_path / "profiling.toml"
    path.write_text(text)
    return path


def test_toml_sub_tables_flatten_to_flat_settings(tmp_path):
    path = _write_toml(
        tmp_path,
        """
        [profiling]
        file_path = "run.h5"

        [profiling.hdf5]
        compression = "gzip"
        compression_level = 4

        [profiling.memray]
        enabled = true
        """,
    )

    assert load_profiling_config(path) == {
        "file_path": "run.h5",
        "hdf5_compression": "gzip",
        "hdf5_compression_level": 4,
        "use_memray": True,
    }


def test_toml_group_and_flat_spelling_of_one_setting_conflict(tmp_path):
    path = _write_toml(
        tmp_path,
        """
        [profiling]
        hdf5_compression = "lzf"

        [profiling.hdf5]
        compression = "gzip"
        """,
    )

    with pytest.raises(ValueError, match="both directly and as hdf5.compression"):
        load_profiling_config(path)


def test_toml_rejects_an_unknown_key_inside_a_group(tmp_path):
    path = _write_toml(tmp_path, '[profiling.hdf5]\ncompressionn = "gzip"\n')

    with pytest.raises(ValueError, match=r"Unknown \[hdf5\] profiling setting"):
        load_profiling_config(path)


def test_toml_rejects_a_group_that_is_not_a_table(tmp_path):
    path = _write_toml(tmp_path, '[profiling]\nhdf5 = "gzip"\n')

    with pytest.raises(ValueError, match="must be a TOML table"):
        load_profiling_config(path)


def test_toml_group_settings_reach_the_config(tmp_path):
    path = _write_toml(
        tmp_path,
        "[profiling]\ndeactivate_file_output = true\n\n[profiling.gpu]\nnvtx = true\n",
    )

    ProfileManager.setup(config_path=path)

    assert ProfileManager.get_config().use_nvtx is True


# --- setup() keyword handling ----------------------------------------------


def test_setup_options_typed_dict_matches_the_declared_settings():
    """The one place a setting is declared is ``ProfilingOptions``.

    ``SetupOptions`` exists only to give type checkers the ``**overrides``
    keyword names, so it has to be kept equal to the dataclass by hand. This
    is what makes that drift a build failure rather than a silent gap.
    """
    assert set(SetupOptions.__annotations__) == set(_CONFIG_FIELDS)


def test_an_unknown_setup_keyword_names_the_closest_match():
    with pytest.raises(TypeError, match=r"'use_likwd' \(did you mean 'use_likwid'\?\)"):
        ProfileManager.setup(use_likwd=True)


def test_an_unknown_setup_keyword_with_no_near_match_still_raises():
    with pytest.raises(TypeError, match="Unknown profiling setting"):
        ProfileManager.setup(zzzzzzzz=True)


def test_precedence_runs_config_path_then_options_then_keywords(tmp_path):
    path = _write_toml(
        tmp_path,
        '[profiling]\nlabel = "from-toml"\nbuffer_limit = 111\noutput_mode = "direct"\n',
    )

    ProfileManager.setup(
        config_path=path,
        options=ProfilingOptions(label="from-options", buffer_limit=222),
        label="from-keyword",
        deactivate_file_output=True,
    )
    config = ProfileManager.get_config()

    assert config.label == "from-keyword"  # keyword beats options
    assert config.buffer_limit == 222  # options beats the TOML file
    assert config.output_mode == "direct"  # only the TOML file set it


# --- load() -----------------------------------------------------------------


@pytest.fixture
def results(tmp_path):
    ProfileManager.setup(file_path=str(tmp_path / "run.h5"))
    with ProfileManager.region("solve"):
        sum(range(100))
    return ProfileManager.finalize(verbose=False, return_results=True)


def test_load_reads_hdf5(tmp_path, results):
    from scope_profiler import load

    assert "solve" in load(tmp_path / "run.h5").region_names


def test_load_reads_json_and_gzipped_json(tmp_path, results):
    from scope_profiler import load, write_profile

    write_profile(results, tmp_path / "run.json")
    write_profile(results, tmp_path / "run.json.gz")

    assert "solve" in load(tmp_path / "run.json").region_names
    assert "solve" in load(tmp_path / "run.json.gz").region_names


def test_load_goes_by_contents_not_by_name(tmp_path, results):
    """A JSON profile under a ``.h5`` name still reads."""
    from scope_profiler import load, write_profile

    written = write_profile(results, tmp_path / "run.json")
    misnamed = tmp_path / "misnamed.h5"
    misnamed.write_bytes(written.read_bytes())

    assert "solve" in load(misnamed).region_names


def test_load_refuses_an_html_report(tmp_path, results):
    from scope_profiler import load, write_profile

    write_profile(results, tmp_path / "report.html")

    with pytest.raises(ValueError, match="write-only"):
        load(tmp_path / "report.html")


def test_load_reports_a_missing_file(tmp_path):
    from scope_profiler import load

    with pytest.raises(FileNotFoundError, match="No profile at"):
        load(tmp_path / "absent.h5")


def test_sniff_recognises_each_format(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 8)
    (tmp_path / "b.bin").write_text(json.dumps({"regions": {}}))
    (tmp_path / "c.bin").write_bytes(gzip.compress(b"{}"))
    (tmp_path / "d.bin").write_text("<!doctype html>")

    assert sniff_profile_format(tmp_path / "a.bin") == FORMAT_HDF5
    assert sniff_profile_format(tmp_path / "b.bin") == FORMAT_JSON
    assert sniff_profile_format(tmp_path / "c.bin") == FORMAT_JSON
    assert sniff_profile_format(tmp_path / "d.bin") == FORMAT_HTML


def test_sniff_falls_back_to_the_file_name(tmp_path):
    """Unrecognisable leading bytes leave the name as the only evidence."""
    (tmp_path / "mystery.json").write_bytes(b"garbage!")
    (tmp_path / "mystery.dat").write_bytes(b"garbage!")

    assert sniff_profile_format(tmp_path / "mystery.json") == FORMAT_JSON
    assert sniff_profile_format(tmp_path / "mystery.dat") == FORMAT_HDF5


def test_sniff_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        sniff_profile_format(tmp_path / "absent.h5")


def test_session_forwards_options_and_config_path(tmp_path):
    path = _write_toml(tmp_path, "[profiling]\nbuffer_limit = 333\n")

    with ProfileManager.session(
        ProfilingOptions(label="from-options"),
        config_path=path,
        file_path=str(tmp_path / "run.h5"),
        verbose=False,
    ):
        config = ProfileManager.get_config()

    assert config.label == "from-options"
    assert config.buffer_limit == 333
    assert config.file_path == str(tmp_path / "run.h5")


def test_session_rejects_an_unknown_setting():
    with pytest.raises(TypeError, match="Unknown profiling setting"):
        with ProfileManager.session(file_pth="run.h5"):
            pass
