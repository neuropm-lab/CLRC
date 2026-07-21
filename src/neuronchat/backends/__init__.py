"""Compute backend factory."""

from __future__ import annotations

from neuronchat.backends.base import ComputeBackend


def get_backend(device: str | list[str] = "cpu") -> ComputeBackend:
    """Create a compute backend.

    Parameters
    ----------
    device : str | list[str]
        ``'cpu'`` for NumPy backend, ``'cuda'`` (auto-detect all GPUs),
        ``'cuda:N'`` (single GPU), or a list of CUDA device strings
        (explicit multi-GPU) for PyTorch GPU backend.
    """
    if isinstance(device, list):
        from neuronchat.backends.torch_backend import TorchBackend

        return TorchBackend(device=device)
    elif device == "cpu":
        from neuronchat.backends.numpy_backend import NumpyBackend

        return NumpyBackend()
    elif device.startswith("cuda"):
        from neuronchat.backends.torch_backend import TorchBackend

        return TorchBackend(device=device)
    else:
        raise ValueError(
            f"Unknown device: '{device}'. "
            "Choose 'cpu', 'cuda', 'cuda:N', or a list of CUDA devices."
        )
