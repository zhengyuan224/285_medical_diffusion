"""Compare two generated sample images (e.g., from latest DDPM checkpoints).

This is useful to see how much the model output changed between checkpoints.

Usage examples:
  # compare the two most recent sample-*.jpg files
  python evaluation/compare_samples.py --samples_dir results/DEFAULT/own_dataset --latest

  # compare two explicit files
  python evaluation/compare_samples.py \
    --file1 results/DEFAULT/own_dataset/sample-21.jpg \
    --file2 results/DEFAULT/own_dataset/sample-22.jpg

Outputs:
- SSIM (structural similarity)
- (optional) LPIPS if torchvision + lpips weights are available
"""

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path when running this file directly.
# When running `python evaluation/compare_samples.py`, Python adds the
# `evaluation/` folder to sys.path, so `import evaluation` would fail.
proj_root = Path(__file__).resolve().parents[1]
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

import numpy as np
import torch
from PIL import Image

from evaluation.pytorch_ssim import ssim


def load_image(path: Path, as_gray: bool = True) -> torch.Tensor:
    img = Image.open(str(path)).convert('RGB' if not as_gray else 'L')
    arr = np.array(img).astype(np.float32) / 255.0
    if as_gray:
        arr = arr[None, ...]  # [1, H, W]
    else:
        arr = arr.transpose(2, 0, 1)  # [C, H, W]
    tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, C, H, W]
    return tensor


def find_latest_samples(folder: Path, pattern: str = "sample-*.jpg"):
    all_files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime)
    if len(all_files) < 2:
        return None, None
    return all_files[-2], all_files[-1]


def main():
    parser = argparse.ArgumentParser(description="Compare two generated sample images using SSIM (+ optional LPIPS).")
    parser.add_argument("--samples_dir", type=str, default=None,
                        help="Directory containing sample-*.jpg outputs")
    parser.add_argument("--file1", type=str, default=None, help="First sample image path")
    parser.add_argument("--file2", type=str, default=None, help="Second sample image path")
    parser.add_argument("--latest", action="store_true",
                        help="Automatically compare the two most recent sample-*.jpg files in --samples_dir")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    args = parser.parse_args()

    if args.latest:
        if args.samples_dir is None:
            raise SystemExit("--samples_dir is required when using --latest")
        f1, f2 = find_latest_samples(Path(args.samples_dir))
        if f1 is None:
            raise SystemExit(f"Not enough sample images found in {args.samples_dir}")
        print(f"Comparing latest two samples:\n  {f1}\n  {f2}")
    else:
        if args.file1 is None or args.file2 is None:
            raise SystemExit("Please provide --file1 and --file2 (or use --latest)")
        f1 = Path(args.file1)
        f2 = Path(args.file2)

    img1 = load_image(f1, as_gray=True).to(args.device)
    img2 = load_image(f2, as_gray=True).to(args.device)

    if img1.shape != img2.shape:
        # crop to minimum common shape
        min_h = min(img1.shape[-2], img2.shape[-2])
        min_w = min(img1.shape[-1], img2.shape[-1])
        img1 = img1[..., :min_h, :min_w]
        img2 = img2[..., :min_h, :min_w]

    with torch.no_grad():
        s = ssim(img1, img2)

    print(f"SSIM between images: {s.item():.6f}")

    try:
        from vq_gan_3d.model.lpips import LPIPS

        lpips = LPIPS().to(args.device)
        with torch.no_grad():
            # LPIPS expects 3-channel, so convert grayscale to 3-channel
            img1_3 = img1.repeat(1, 3, 1, 1)
            img2_3 = img2.repeat(1, 3, 1, 1)
            lp = lpips(img1_3, img2_3)
        print(f"LPIPS between images: {lp.item():.6f}")
    except Exception as e:
        print(f"LPIPS unavailable: {e}")


if __name__ == "__main__":
    main()
