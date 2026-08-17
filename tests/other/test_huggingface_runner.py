from types import SimpleNamespace

import pytest
import torch

import huggingface_test_runner
from huggingface_test_runner import load_huggingface_reference, to_torch_type


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("float16", torch.float16),
        ("float32", torch.float32),
        ("bfloat16", torch.bfloat16),
    ],
)
def test_to_torch_type(name, expected):
    assert to_torch_type(name) is expected


def test_to_torch_type_rejects_unsupported_type():
    with pytest.raises(ValueError, match="Unsupported HuggingFace tensor type"):
        to_torch_type("float8")


def test_reference_model_uses_configured_tensor_type(monkeypatch):
    calls = []

    class FakeModel:
        def __init__(self):
            self.eval_called = False

        def eval(self):
            self.eval_called = True
            return self

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_path, **kwargs):
            calls.append((model_path, kwargs))
            return FakeModel()

    monkeypatch.setattr(
        huggingface_test_runner, "AutoModelForCausalLM", FakeAutoModel)
    config = SimpleNamespace(torch_dtype=torch.float32)

    model = load_huggingface_reference("model", config, "bfloat16")

    assert config.torch_dtype is torch.bfloat16
    assert model.eval_called
    assert calls == [
        (
            "model",
            {
                "config": config,
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
                "trust_remote_code": True,
            },
        )
    ]
