import pytest

from crazeai.data import make_supervised_features, normalize_record
from scripts.prepare_dataset import HF_MIX, canonical_hash


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        if add_generation_prompt:
            return [101, 102, 103]
        return [101, 102, 103, 201, 202, 203]


def test_normalize_legacy_instruction_schema():
    messages = normalize_record(
        {
            "instruction": "Write a Python function.",
            "input": "It should add two numbers.",
            "output": "def add(a, b):\n    return a + b",
        }
    )
    assert messages is not None
    assert messages[-2]["role"] == "user"
    assert "It should add two numbers." in messages[-2]["content"]
    assert messages[-1]["role"] == "assistant"
    assert messages[0]["role"] == "system"


def test_rejects_missing_assistant_answer():
    assert normalize_record({"messages": [{"role": "user", "content": "Hi"}]}) is None


def test_supervision_masks_prompt_and_trains_answer_only():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    features = make_supervised_features(messages, FakeTokenizer(), max_length=128)
    assert features is not None
    assert features["input_ids"] == [101, 102, 103, 201, 202, 203]
    assert features["labels"] == [-100, -100, -100, 201, 202, 203]
    assert len(features["attention_mask"]) == len(features["input_ids"])


def test_supervision_refuses_non_prefix_tokenization():
    class BrokenTokenizer:
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
            return [1, 2, 3] if add_generation_prompt else [9, 8, 7, 6]

    with pytest.raises(ValueError, match="does not preserve the prompt prefix"):
        make_supervised_features(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
            BrokenTokenizer(),
            max_length=128,
        )


def test_dataset_mix_is_exact():
    assert sum(count for _, count in HF_MIX) == 49_988


def test_canonical_hash_is_stable():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    assert canonical_hash(messages) == canonical_hash(messages)
    assert canonical_hash(messages) != canonical_hash(list(reversed(messages)))
