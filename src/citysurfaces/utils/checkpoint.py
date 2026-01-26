import os
import shutil
from typing import Any, Dict, Optional

import torch


def safe_save_checkpoint(state: Dict[str, Any], path: str, max_keep: int = 3) -> None:
    """
    Safely save a checkpoint by writing to a temp file and renaming.
    Optionally rotate older checkpoints (simple numeric suffix, newest kept).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)
    if max_keep and max_keep > 0:
        _rotate_checkpoints(os.path.dirname(path) or ".", os.path.basename(path), max_keep)


def _rotate_checkpoints(dirpath: str, basename: str, max_keep: int) -> None:
    """
    Keep at most max_keep checkpoints that share the same basename and numeric suffix.
    """
    stem, ext = os.path.splitext(basename)
    def key(p):
        name = os.path.basename(p)
        if name == basename:
            return 0
        try:
            suffix = name.replace(stem + "_", "").replace(ext, "")
            return int(suffix)
        except ValueError:
            return float("inf")

    candidates = [os.path.join(dirpath, f) for f in os.listdir(dirpath) if f.startswith(stem) and f.endswith(ext)]
    candidates = sorted(candidates, key=os.path.getmtime, reverse=True)
    for stale in candidates[max_keep:]:
        try:
            os.remove(stale)
        except OSError:
            pass
