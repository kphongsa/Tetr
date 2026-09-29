"""Checkpoints: save everything needed to continue training later, crash-safely.

A checkpoint is one dict written with torch.save (PyTorch's file format; it
stores network weights efficiently and also plain Python data). What goes
in it is decided by the training loop and the learner; this module only
makes writing and reading it safe.

Atomic write: we save to "<name>.tmp" first and then rename it over the real
file. A rename is all-or-nothing on every OS we care about, so if the
process dies mid-save (Ctrl+C, power loss), the old checkpoint is still
intact instead of half-overwritten.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic rename (overwrites the old file)


def load_checkpoint(path: Path) -> dict[str, Any]:
    # weights_only=True: only load tensors and plain Python data, never
    # arbitrary objects. A checkpoint file can't run code on load this way.
    return torch.load(Path(path), weights_only=True)
