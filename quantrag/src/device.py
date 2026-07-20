"""
src/device.py

Universal hardware detection for QuantRAG.
Auto-selects the best available compute device on ANY machine and decides
whether parallel embedding is safe on that hardware.

    CUDA (NVIDIA GPU)  →  fastest — parallel embedding ON
    MPS  (Apple GPU)   →  fast    — parallel embedding OFF (spawn unstable)
    CPU                →  universal fallback — parallel via process pool

The same codebase runs optimally on a cloud GPU, a MacBook, a Windows GPU
laptop, or a CPU-only CI runner — no code changes needed.
"""

import os
from rich.console import Console

console = Console()


def get_optimal_device() -> str:
    """
    Detect and return the best available torch device string.
    Returns one of: "cuda" | "mps" | "cpu"
    """
    import torch

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        console.print(f"[green]  ✓ Using CUDA GPU: {name}[/green]")
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        console.print("[green]  ✓ Using Apple MPS (Metal GPU)[/green]")
        return "mps"

    console.print("[yellow]  ✓ Using CPU (no GPU detected)[/yellow]")
    return "cpu"


def get_optimal_batch_size(device: str) -> int:
    """
    Return a safe, efficient embedding batch size for the given device.
        CUDA — large batches, dedicated VRAM handles it
        MPS  — moderate batches, shared memory needs headroom
        CPU  — small batches, no benefit from large ones
    """
    return {
        "cuda": 128,
        "mps":  16,
        "cpu":  32,
    }.get(device, 32)


def should_parallel_embed(device: str, mode: str = "auto") -> tuple:
    """
    Decide whether to use parallel (multi-worker) embedding.

    Parallelism helps a lot on CUDA and multi-core CPU, but the
    ProcessPoolExecutor + spawn approach is unstable with PyTorch MPS
    on macOS (causes the "MPS backend out of memory" crash). So on MPS
    we run single-stream — still fast, always stable.

    Args:
        device: "cuda" | "mps" | "cpu"
        mode:   "auto" | "on" | "off"  (override the auto-decision)

    Returns:
        (parallel_enabled: bool, num_workers: int)
    """
    if mode == "off":
        return False, 1

    if mode == "on":
        # User forces parallel — pick a sensible worker count
        workers = {"cuda": 2, "mps": 1, "cpu": max(1, (os.cpu_count() or 2) // 2)}
        return (workers.get(device, 1) > 1), workers.get(device, 1)

    # mode == "auto" — the recommended default
    if device == "cuda":
        # CUDA handles parallel embedding streams well
        return True, 2
    if device == "cpu":
        # CPU benefits from process-level parallelism if cores available
        cores = os.cpu_count() or 2
        workers = max(1, cores // 2)
        return (workers > 1), workers
    # MPS — single-stream for stability
    return False, 1


def configure_device_memory(device: str) -> None:
    """
    Apply device-specific memory settings.

    MPS:  high-watermark ratio 0.0 lets Apple GPU spill to system RAM
          instead of crashing with "MPS backend out of memory".
    CUDA: expandable segments reduces fragmentation on long runs.
    All:  disable tokenizer thread contention.

    Safe to call for any device — only the relevant vars are set.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if device == "mps":
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    elif device == "cuda":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def setup_compute(parallel_mode: str = "auto") -> dict:
    """
    One-call setup — configures environment and returns a config dict.

    Call once at the start of any script that does embedding.

    Args:
        parallel_mode: "auto" | "on" | "off"

    Returns:
        {
          "device":       "cuda" | "mps" | "cpu",
          "batch_size":   int,
          "parallel":     bool,
          "num_workers":  int,
        }
    """
    device = get_optimal_device()
    configure_device_memory(device)

    batch_size          = get_optimal_batch_size(device)
    parallel, n_workers = should_parallel_embed(device, parallel_mode)

    console.print(
        f"[dim]  Device: {device} | Batch: {batch_size} | "
        f"Parallel: {parallel} ({n_workers} workers)[/dim]"
    )

    return {
        "device":      device,
        "batch_size":  batch_size,
        "parallel":    parallel,
        "num_workers": n_workers,
    }