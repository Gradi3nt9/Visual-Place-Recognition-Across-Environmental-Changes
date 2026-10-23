"""Metadata recorded with every run so each reported number can be reproduced."""
from __future__ import annotations

import platform
import random
import subprocess
from datetime import datetime, timezone

import numpy as np

from .config import REPO_ROOT


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_metadata() -> dict:
    status = _git("status", "--porcelain")
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        import torch
        meta["torch"] = torch.__version__
        meta["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    return meta
