import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from scripts.train import TokenizedJsonlDataset, encode_compact  # noqa: E402


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        if add_generation_prompt:
            return [101, 102, 103]
        return [101, 102, 103, 201, 202, 203]


def test_encode_compact_marks_prompt_length():
    row = {"instruction": "question", "output": "answer"}
    record = encode_compact(row, FakeTokenizer(), max_length=128)
    assert record == {"i": [101, 102, 103, 201, 202, 203], "p": 3}


def test_encode_compact_rejects_invalid_rows():
    assert encode_compact({"messages": [{"role": "user", "content": "hi"}]}, FakeTokenizer(), 128) is None


def test_dataset_rebuilds_labels_and_mask(tmp_path):
    path = tmp_path / "tokenized.jsonl"
    rows = [
        {"i": [1, 2, 3, 4, 5], "p": 2},
        {"i": [9, 8, 7], "p": 1},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    dataset = TokenizedJsonlDataset(path)
    assert len(dataset) == 2
    first = dataset[0]
    assert first["input_ids"] == [1, 2, 3, 4, 5]
    assert first["attention_mask"] == [1] * 5
    assert first["labels"] == [-100, -100, 3, 4, 5]
    assert dataset[-1]["labels"] == [-100, 8, 7]
