import pytest

from scope_profiler import ProfileManager
from scope_profiler.profile_config import load_profiling_config


def test_setup_loads_profiling_toml_and_direct_values_override(tmp_path):
    config_path = tmp_path / "profiling.toml"
    config_path.write_text(
        "[profiling]\n"
        "use_nvtx = true\n"
        "use_gpu_timing = true\n"
        "gpu_timing_backend = 'cupy'\n"
        "buffer_limit = 2048\n"
        "label = 'from-file'\n",
        encoding="utf-8",
    )

    ProfileManager.setup(
        config_path=config_path, buffer_limit=4096, use_gpu_timing=False
    )
    config = ProfileManager.get_config()

    assert config.use_nvtx is True
    assert config.use_gpu_timing is False
    assert config.gpu_timing_backend == "cupy"
    assert config.buffer_limit == 4096
    assert config.label == "from-file"


def test_load_profiling_config_rejects_unknown_settings(tmp_path):
    config_path = tmp_path / "profiling.toml"
    config_path.write_text("[profiling]\nnot_a_setting = true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown profiling setting"):
        load_profiling_config(config_path)


def test_load_profiling_config_accepts_top_level_settings(tmp_path):
    config_path = tmp_path / "profiling.toml"
    config_path.write_text("buffer_limit = 7\n", encoding="utf-8")

    assert load_profiling_config(config_path) == {"buffer_limit": 7}


def test_output_mode_can_be_configured_from_toml(tmp_path):
    config_path = tmp_path / "profiling.toml"
    config_path.write_text("[profiling]\noutput_mode = 'direct'\n", encoding="utf-8")

    ProfileManager.setup(config_path=config_path)

    assert ProfileManager.get_config().output_mode == "direct"


def test_invalid_output_mode_is_rejected():
    with pytest.raises(ValueError, match="output_mode"):
        ProfileManager.setup(output_mode="sharded")


def test_hdf5_storage_settings_load_from_toml(tmp_path):
    config_path = tmp_path / "profiling.toml"
    config_path.write_text(
        "[profiling]\n"
        "hdf5_compression = 'gzip'\n"
        "hdf5_compression_level = 6\n"
        "hdf5_chunk_size = 4096\n",
        encoding="utf-8",
    )

    ProfileManager.setup(config_path=config_path)
    config = ProfileManager.get_config()

    assert config.hdf5_compression == "gzip"
    assert config.hdf5_compression_level == 6
    assert config.hdf5_chunk_size == 4096


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hdf5_compression": "bzip2"}, "hdf5_compression"),
        ({"hdf5_compression": "gzip", "hdf5_compression_level": 10}, "GZIP"),
        ({"hdf5_compression": "lzf", "hdf5_compression_level": 1}, "LZF"),
        ({"hdf5_chunk_size": 0}, "chunk_size"),
    ],
)
def test_invalid_hdf5_storage_settings_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ProfileManager.setup(**kwargs)
