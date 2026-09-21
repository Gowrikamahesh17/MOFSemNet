import pytest
import torch

from src.common.config import RuntimeConfig
from src.common.runtime import get_device


def test_auto_resolves_to_available_device():
    device = get_device(RuntimeConfig(device="auto"))
    assert device.type in ("cuda", "mps", "cpu")


def test_cpu_always_available():
    assert get_device(RuntimeConfig(device="cpu")).type == "cpu"


def test_none_config_defaults_to_auto():
    device = get_device(None)
    assert device.type in ("cuda", "mps", "cpu")


def test_rejects_unknown_device_choice():
    with pytest.raises(ValueError, match="device must be one of"):
        get_device(RuntimeConfig(device="tpu"))


def test_cuda_request_raises_when_unavailable():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this machine; the failure path isn't exercised here.")
    with pytest.raises(RuntimeError, match="cuda"):
        get_device(RuntimeConfig(device="cuda"))
