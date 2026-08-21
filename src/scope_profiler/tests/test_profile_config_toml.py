import pytest

from scope_profiler import ProfileManager
from scope_profiler.profile_config import load_profiling_config


def test_setup_loads_profiling_toml_and_direct_values_override(tmp_path):
    config_path = tmp_path / "profiling.toml"
    config_path.write_text(
        "[profiling]\nuse_nvtx = true\nbuffer_limit = 2048\nlabel = 'from-file'\n",
        encoding="utf-8",
    )

    ProfileManager.setup(config_path=config_path, buffer_limit=4096)
    config = ProfileManager.get_config()

    assert config.use_nvtx is True
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
