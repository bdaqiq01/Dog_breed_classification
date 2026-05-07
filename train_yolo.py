"""Standalone YOLO11s trainer for the dog-breed project.

Mirrors the training + test-evaluation cells from model_develpment.ipynb so
the run can be detached from a notebook kernel and survive SSH disconnects.

Usage (from the project root, with the venv activated):
    python train_yolo.py

Or, detached so it survives logout:
    tmux new -s yolo
    source venv/bin/activate
    python train_yolo.py 2>&1 | tee runs/detect/train.log
    # Detach with Ctrl+B then D. Reattach later with: tmux attach -t yolo
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ultralytics defaults to ~/.config/Ultralytics (home quota); use project storage instead.
PROJECT_ROOT = Path(__file__).resolve().parent


def proj_rel(path: str | Path, root: Path = PROJECT_ROOT) -> str:
    """Path under project for logs (handles cwd vs realpath mismatch on shared storage)."""
    path = Path(path).resolve()
    bases: list[Path] = []
    seen: set[str] = set()
    for b in (Path(root).resolve(), Path.cwd().resolve()):
        k = str(b)
        if k not in seen:
            seen.add(k)
            bases.append(b)
    for base in bases:
        try:
            return str(path.relative_to(base))
        except ValueError:
            continue
    proj = "Dog_breed_classification"
    parts = path.parts
    if proj in parts:
        idx = parts.index(proj)
        tail = parts[idx + 1 :]
        return str(Path(*tail)) if tail else "."
    return path.name


_ULTRA_PARENT = PROJECT_ROOT / ".cache"
_ULTRA_PARENT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ULTRA_PARENT))

import torch
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

YOLO_ROOT = PROJECT_ROOT / "data" / "yolo_dataset"
YOLO_DATA_YAML = YOLO_ROOT / "dogs.yaml"
YOLO_RUNS_DIR = PROJECT_ROOT / "runs" / "detect"
YOLO_RUNS_DIR.mkdir(parents=True, exist_ok=True)
YOLO_MODEL_NAME = "yolo11s.pt"
SEED = 42

SETTINGS.update({
    "runs_dir":     str(PROJECT_ROOT / "runs"),
    "datasets_dir": str(PROJECT_ROOT / "data"),
    "weights_dir":  str(PROJECT_ROOT / "weights"),
})


def pick_device() -> int | str:
    """Use CUDA only if cuDNN can actually run a conv; otherwise fall back to CPU."""
    if not torch.cuda.is_available():
        return "cpu"
    try:
        x = torch.zeros(1, 3, 32, 32, device="cuda")
        w = torch.zeros(8, 3, 3, 3, device="cuda")
        torch.nn.functional.conv2d(x, w)
        torch.cuda.synchronize()
        return 0
    except Exception as e:  # pragma: no cover - environment-dependent
        print(
            f"[YOLO] CUDA available but cuDNN conv failed ({e.__class__.__name__}); "
            f"falling back to CPU.",
            flush=True,
        )
        return "cpu"


def main() -> int:
    if not YOLO_DATA_YAML.is_file():
        print(f"ERROR: missing {proj_rel(YOLO_DATA_YAML)}. Run the dataset-prep cell first.", file=sys.stderr)
        return 1

    device = pick_device()
    print("[YOLO] Project root  : .", flush=True)
    print(f"[YOLO] Data yaml     : {proj_rel(YOLO_DATA_YAML)}", flush=True)
    print(f"[YOLO] Runs dir      : {proj_rel(YOLO_RUNS_DIR)}", flush=True)
    print(f"[YOLO] Training device: {device}", flush=True)

    model = YOLO(YOLO_MODEL_NAME)
    results = model.train(
        data=str(YOLO_DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=16,
        patience=25,
        project=str(YOLO_RUNS_DIR),
        name="dogbreeds_yolo11s",
        seed=SEED,
        device=device,
        amp=(device != "cpu"),
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"[YOLO] Best weights  : {proj_rel(best)}", flush=True)

    print("[YOLO] Evaluating best weights on the held-out test split...", flush=True)
    best_model = YOLO(str(best))
    test_metrics = best_model.val(
        data=str(YOLO_DATA_YAML),
        split="test",
        imgsz=640,
        batch=16,
        project=str(YOLO_RUNS_DIR),
        name="dogbreeds_yolo11s_test",
        device=device,
    )
    print(f"[YOLO] test mAP@0.5      : {test_metrics.box.map50:.4f}", flush=True)
    print(f"[YOLO] test mAP@0.5:0.95 : {test_metrics.box.map:.4f}", flush=True)
    print(f"[YOLO] test precision    : {test_metrics.box.mp:.4f}", flush=True)
    print(f"[YOLO] test recall       : {test_metrics.box.mr:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
