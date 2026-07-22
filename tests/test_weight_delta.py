"""
Tests for the int8 weight-delta mechanism (src/visual_module/visual_classifier.py).

The delta files are how fine-tuned models are reproduced exactly, so the
round-trip must be faithful (within int8 quantisation error) and applying a
delta to the wrong base model must fail loudly (GAPS.md #1).
"""

import pytest
import torch
import torch.nn as nn

from src.visual_module.visual_classifier import get_delta_base_model, load_weight_delta


def _make_delta_checkpoint(base: nn.Module, fine_tuned: nn.Module, base_model_name: str):
    """Builds a checkpoint in the exact format save_weight_delta() writes."""
    delta = {}
    base_state = base.state_dict()
    for key, ft_param in fine_tuned.state_dict().items():
        diff = ft_param.float() - base_state[key].float()
        max_abs = diff.abs().max().item()
        if max_abs < 1e-9:
            continue
        scale = max_abs / 127.0
        quant = (diff / scale).round().clamp(-127, 127).to(torch.int8)
        delta[key] = {"q": quant, "s": scale}
    return {"base_model": base_model_name, "dtype": "int8", "delta": delta}


class _NamedModel(nn.Module):
    """Tiny stand-in for a HF model exposing name_or_path."""

    def __init__(self, name_or_path):
        super().__init__()
        self.name_or_path = name_or_path
        self.linear = nn.Linear(8, 4)


@pytest.fixture
def delta_file(tmp_path):
    torch.manual_seed(42)
    base = _NamedModel("test/base-model")
    fine_tuned = _NamedModel("test/base-model")
    checkpoint = _make_delta_checkpoint(base, fine_tuned, "test/base-model")
    path = tmp_path / "delta.pt"
    torch.save(checkpoint, path)
    return base, fine_tuned, str(path)


class TestRoundTrip:
    def test_delta_reconstructs_fine_tuned_weights(self, delta_file):
        base, fine_tuned, path = delta_file
        expected = {k: v.float().clone() for k, v in fine_tuned.state_dict().items()}
        # int8 quantisation error bound: one quantisation step per tensor
        tolerances = {
            key: max(
                (ft - base.state_dict()[key].float()).abs().max().item() / 127.0,
                1e-6,
            )
            for key, ft in expected.items()
        }

        load_weight_delta(base, path)

        for key, ft in expected.items():
            assert torch.allclose(
                base.state_dict()[key].float(), ft, atol=tolerances[key]
            ), f"Round-trip mismatch on {key}"

    def test_get_delta_base_model_reads_checkpoint(self, delta_file):
        _, _, path = delta_file
        assert get_delta_base_model(path) == "test/base-model"


class TestBaseMismatchGuard:
    def test_wrong_base_raises(self, delta_file, tmp_path):
        _, _, path = delta_file
        wrong_base = _NamedModel("test/OTHER-model")
        with pytest.raises(ValueError, match="test/base-model"):
            load_weight_delta(wrong_base, path)

    def test_wrong_base_with_force_applies(self, delta_file):
        _, _, path = delta_file
        wrong_base = _NamedModel("test/OTHER-model")
        load_weight_delta(wrong_base, path, force=True)  # must not raise

    def test_matching_base_applies_cleanly(self, delta_file):
        base, _, path = delta_file
        load_weight_delta(base, path)  # must not raise

    def test_unverifiable_model_identity_raises(self, delta_file):
        _, _, path = delta_file
        anonymous = nn.Linear(8, 4)  # exposes neither name_or_path nor config
        with pytest.raises(ValueError, match="name_or_path"):
            load_weight_delta(anonymous, path)

    def test_unverifiable_model_identity_with_force_applies(self, delta_file):
        _, _, path = delta_file
        anonymous = _NamedModel("x")
        anonymous.name_or_path = None
        load_weight_delta(anonymous, path, force=True)  # must not raise
