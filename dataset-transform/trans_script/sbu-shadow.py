"""Convert SBU-shadow (0/255 masks) into YOLO26-sem index PNGs + data.yaml."""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MASK_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}

SPLITS = (
    ("SBU-Train", "train"),
    ("SBU-Test", "val"),
)

_REPO = Path(__file__).resolve().parents[2]


def _index_stems(folder: Path, suffixes: set[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not folder.is_dir():
        return out
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in suffixes:
            out[p.stem] = p
    return out


def _link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def _convert_mask(src: Path, dst: Path) -> np.ndarray:
    raw = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise SystemExit(f"Failed to read mask: {src}")
    yolo = (raw > 127).astype(np.uint8)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dst), yolo):
        raise SystemExit(f"Failed to write mask: {dst}")
    return yolo


def _write_data_yaml(dst: Path) -> None:
    yolo_lab = _REPO / "yolo-lab"
    try:
        root = dst.resolve().relative_to(yolo_lab.resolve()).as_posix()
    except ValueError:
        root = dst.resolve().as_posix()
    text = (
        f"path: {root}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"masks_dir: masks\n"
        f"names:\n"
        f"  0: bright\n"
        f"  1: dark\n"
    )
    (dst / "data.yaml").write_text(text, encoding="utf-8")


def convert_split(src_root: Path, dst_root: Path, src_name: str, split: str) -> dict[str, int]:
    img_dir = src_root / src_name / "ShadowImages"
    mask_dir = src_root / src_name / "ShadowMasks"
    if not img_dir.is_dir() or not mask_dir.is_dir():
        raise SystemExit(f"Missing ShadowImages or ShadowMasks under {src_root / src_name}")

    images = _index_stems(img_dir, IMAGE_SUFFIXES)
    masks = _index_stems(mask_dir, MASK_SUFFIXES)
    stems = sorted(set(images) & set(masks))
    skipped_img = len(set(images) - set(masks))
    skipped_mask = len(set(masks) - set(images))

    n_link = n_copy = 0
    unique_ok = 0
    sample_unique: list[str] = []
    for stem in stems:
        kind = _link_or_copy(images[stem], dst_root / "images" / split / images[stem].name)
        if kind == "link":
            n_link += 1
        else:
            n_copy += 1
        yolo = _convert_mask(masks[stem], dst_root / "masks" / split / f"{stem}.png")
        vals = set(int(v) for v in np.unique(yolo))
        if vals <= {0, 1}:
            unique_ok += 1
        if len(sample_unique) < 3:
            sample_unique.append(f"{split}/{stem}.png unique={sorted(vals)}")

    return {
        "paired": len(stems),
        "skip_image_only": skipped_img,
        "skip_mask_only": skipped_mask,
        "link": n_link,
        "copy": n_copy,
        "mask_01": unique_ok,
        "samples": sample_unique,  # type: ignore[dict-item]
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert SBU-shadow to YOLO26-sem layout")
    p.add_argument(
        "--src",
        default=str(_REPO / "night-iris" / "data" / "SBU-shadow"),
        help="SBU-Train / SBU-Test root (ShadowImages + ShadowMasks)",
    )
    p.add_argument(
        "--dst",
        default=str(_REPO / "yolo-lab" / "datasets" / "SBU-shadow-yolo"),
        help="YOLO-sem dataset output (images/ + masks/ + data.yaml)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    if not src.is_dir():
        raise SystemExit(f"Source not found: {src}")

    dst.mkdir(parents=True, exist_ok=True)
    for src_name, split in SPLITS:
        stats = convert_split(src, dst, src_name, split)
        samples = stats.pop("samples")
        print(f">> {split}: paired={stats['paired']}  link={stats['link']}  copy={stats['copy']}")
        print(
            f"   skip image-only={stats['skip_image_only']}  "
            f"mask-only={stats['skip_mask_only']}  masks_in_01={stats['mask_01']}"
        )
        for line in samples:
            print(f"   sample {line}")
        if stats["paired"] == 0:
            raise SystemExit(f"No paired image/mask in {src / src_name}")
        if stats["mask_01"] != stats["paired"]:
            raise SystemExit("Some converted masks are not strictly {{0,1}}")

    _write_data_yaml(dst)
    print(f">> data.yaml -> {dst.resolve() / 'data.yaml'}")


if __name__ == "__main__":
    main()
