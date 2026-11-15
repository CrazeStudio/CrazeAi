from pathlib import Path

import pytest

from crazeai.runtime import resolve_model_name


def test_missing_export_path_falls_back_to_matching_lora_output(tmp_path: Path):
    adapter_dir = tmp_path / "crazeai-test"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    with pytest.warns(UserWarning, match="using the matching training checkpoint"):
        resolved = resolve_model_name(str(tmp_path / "crazeai-test-export"))

    assert resolved == str(adapter_dir)


def test_training_output_resolves_to_latest_adapter_checkpoint(tmp_path: Path):
    run_dir = tmp_path / "crazeai-test"
    older = run_dir / "checkpoint-10"
    newer = run_dir / "checkpoint-30"
    older.mkdir(parents=True)
    newer.mkdir()
    (older / "adapter_config.json").write_text("{}", encoding="utf-8")
    (newer / "adapter_config.json").write_text("{}", encoding="utf-8")

    assert resolve_model_name(str(run_dir)) == str(newer)


def test_missing_local_path_has_actionable_error_and_nearby_candidates(tmp_path: Path):
    nearby = tmp_path / "valid-model"
    nearby.mkdir()
    (nearby / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Nearby loadable checkpoints"):
        resolve_model_name(str(tmp_path / "missing-model"))


def test_hugging_face_model_id_is_preserved():
    assert resolve_model_name("Qwen/Qwen2.5-0.5B-Instruct") == "Qwen/Qwen2.5-0.5B-Instruct"
