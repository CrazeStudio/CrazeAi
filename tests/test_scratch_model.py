import torch

from crazeai.model import CrazeAIConfig, CrazeAIModel


def test_model_forward_shape_and_causal_attention():
    config = CrazeAIConfig(vocab_size=300, max_seq_len=16, d_model=32, n_layers=2, n_heads=4, dropout=0)
    model = CrazeAIModel(config).eval()
    tokens = torch.randint(0, config.vocab_size, (2, 8))
    logits = model(tokens)
    assert logits.shape == (2, 8, config.vocab_size)

    changed_future = tokens.clone()
    changed_future[:, 7] = (changed_future[:, 7] + 1) % config.vocab_size
    with torch.no_grad():
        original_prefix = model(tokens)[:, :7]
        changed_prefix = model(changed_future)[:, :7]
    assert torch.allclose(original_prefix, changed_prefix, atol=1e-6)


def test_checkpoint_round_trip(tmp_path):
    config = CrazeAIConfig(vocab_size=300, max_seq_len=16, d_model=32, n_layers=2, n_heads=4)
    model = CrazeAIModel(config)
    model.save_pretrained(tmp_path)
    loaded = CrazeAIModel.from_pretrained_dir(tmp_path, torch.device("cpu"))
    assert loaded.config == config
    assert (tmp_path / "config.json").is_file()
    assert (tmp_path / "model.pt").is_file()
