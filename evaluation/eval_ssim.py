"""Compute SSIM between two directories of medical volumes.

Usage:
  python evaluation/eval_ssim.py \
    --ref_dir my_dataset \
    --pred_dir outputs/2026-03-15/your_run \
    --ext .nii.gz

This script will match files by name and compute a per-volume SSIM (3D) using
`evaluation.pytorch_ssim.ssim_3d`.

Notes:
- This is intended for 3D volumes stored as NIfTI (.nii/.nii.gz), but it can also
  handle 2D image tensors.
- If a pair of files have different shapes, the script will crop to the minimum
  overlapping shape along each axis.
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

from evaluation.pytorch_ssim import ssim_3d


def load_volume(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    arr = img.get_fdata(dtype=np.float32)
    return arr


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    """Convert a numpy array to a Torch tensor with shape [1, 1, D, H, W] or [1, C, H, W]."""
    if arr.ndim == 2:
        arr = arr[None, None, ...]
    elif arr.ndim == 3:
        arr = arr[None, None, ...]
    elif arr.ndim == 4:
        # assume (C, H, W, D) or (D, H, W, C) is not common; assume (C, D, H, W)
        arr = arr[None, ...]
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")
    return torch.from_numpy(arr).float()


def crop_to_min(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop two arrays to the minimum overlapping shape along each axis."""
    min_shape = tuple(min(sa, sb) for sa, sb in zip(a.shape, b.shape))
    slices = tuple(slice(0, m) for m in min_shape)
    return a[slices], b[slices]


def main(args):
    ref_dir = Path(args.ref_dir)
    pred_dir = Path(args.pred_dir)

    ref_files = sorted([p for p in ref_dir.glob(f"*{args.ext}") if p.is_file()])
    pred_files = sorted([p for p in pred_dir.glob(f"*{args.ext}") if p.is_file()])

    if len(ref_files) == 0:
        raise SystemExit(f"No files found in ref_dir={ref_dir} with ext={args.ext}")
    if len(pred_files) == 0:
        raise SystemExit(f"No files found in pred_dir={pred_dir} with ext={args.ext}")

    # match by filename
    pred_map = {p.name: p for p in pred_files}
    pairs = []
    for ref_path in ref_files:
        if ref_path.name in pred_map:
            pairs.append((ref_path, pred_map[ref_path.name]))
        elif args.allow_missing:
            continue
        else:
            raise SystemExit(
                f"Missing prediction for reference file: {ref_path.name}. "
                "Make sure the filenames match exactly or use --allow-missing."
            )

    if len(pairs) == 0:
        raise SystemExit("No matching files found between ref_dir and pred_dir.")

    scores = []
    for ref_path, pred_path in tqdm(pairs, desc="Computing SSIM"):
        ref = load_volume(ref_path)
        pred = load_volume(pred_path)

        if ref.shape != pred.shape:
            ref, pred = crop_to_min(ref, pred)

        ref_t = to_tensor(ref).to(args.device)
        pred_t = to_tensor(pred).to(args.device)

        with torch.no_grad():
            score = ssim_3d(ref_t, pred_t).item()
        scores.append(score)

        if args.verbose:
            print(f"{ref_path.name}: SSIM={score:.6f}")

    scores = np.array(scores, dtype=np.float32)
    print("\n----- SSIM Results -----")
    print(f"n_pairs: {len(scores)}")
    print(f"mean: {scores.mean():.6f}")
    print(f"std : {scores.std():.6f}")
    print(f"min : {scores.min():.6f}")
    print(f"max : {scores.max():.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SSIM between two folders of volumes.")
    parser.add_argument("--ref_dir", type=str, required=True, help="Reference / ground truth folder")
    parser.add_argument("--pred_dir", type=str, required=True, help="Predicted folder")
    parser.add_argument("--ext", type=str, default=".nii", help="File extension to match (default: .nii)")
    parser.add_argument("--allow-missing", action="store_true", help="Skip ref files that have no matching pred file")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (cpu or cuda)")
    parser.add_argument("--verbose", action="store_true", help="Print per-case SSIM")
    args = parser.parse_args()

    main(args)
