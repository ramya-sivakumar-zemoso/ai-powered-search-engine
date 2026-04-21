"""Avoid Triton autotuner requiring a GPU driver at import time (Megablocks/STK → Nomic MoE on CPU).

On hosts with CUDA builds of PyTorch but no NVIDIA driver (or no GPU), older Triton
``@triton.autotune`` paths could call into the CUDA driver for benchmarking. When
``Autotuner.__init__`` accepts a ``do_bench`` hook, we inject ``triton.testing.do_bench``
before STK imports. Triton 3.x benches via ``triton.testing`` internally and no longer
takes ``do_bench`` on ``Autotuner``; the shim then becomes a no-op for that API.
"""
from __future__ import annotations

import inspect
from typing import Any, Callable

_done = False


def ensure_triton_cpu_import_safe() -> None:
    """Idempotent: patch Triton autotuner so CPU-only hosts can import MoE dependencies."""
    global _done
    if _done:
        return
    _done = True
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        return
    try:
        import triton.runtime.autotuner as autotuner_mod
        import triton.testing
    except Exception:
        return

    if getattr(autotuner_mod.Autotuner.__init__, "_ai_search_cpu_shim", False):
        return

    _orig: Callable[..., Any] = autotuner_mod.Autotuner.__init__
    _accepts_do_bench = "do_bench" in inspect.signature(_orig).parameters

    def _wrapped(self: Any, *args: Any, **kwargs: Any) -> None:
        # Triton 3.x Autotuner benches via triton.testing.do_bench internally (no do_bench kwarg).
        if _accepts_do_bench and kwargs.get("do_bench") is None:
            kwargs["do_bench"] = lambda kernel_call, quantiles: triton.testing.do_bench(
                kernel_call,
                warmup=25,
                rep=100,
                quantiles=quantiles,
            )
        return _orig(self, *args, **kwargs)

    _wrapped._ai_search_cpu_shim = True  # type: ignore[attr-defined]
    autotuner_mod.Autotuner.__init__ = _wrapped
