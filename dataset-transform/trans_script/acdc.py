"""Convert ACDC adverse-condition images into YOLO datasets.

Night is the default. Semantic masks use Cityscapes trainIds (255 = ignore),
with ACDC invalid pixels forced to 255. Detection labels come from the official
COCO-format instance JSON. That JSON has eight thing classes and does not
include traffic light or traffic sign.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_DOWNLOADS = Path.home() / "Downloads"

CITYSCAPES_19 = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)

# ACDC detection category ids follow Cityscapes instance ids.
DETECT_IDS = {
    24: 0,  # person
    25: 1,  # rider
    26: 2,  # car
    27: 3,  # truck
    28: 4,  # bus
    31: 5,  # train
    32: 6,  # motorcycle
    33: 7,  # bicycle
}
DETECT_NAMES = (
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _read_zip_gray(z: zipfile.ZipFile, name: str) -> np.ndarray:
    raw = z.read(name)
    mask = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise SystemExit(f"Failed to decode {name}")
    return mask


def _yaml_path(dst: Path) -> str:
    yolo_lab = _REPO / "yolo-lab"
    try:
        return dst.resolve().relative_to(yolo_lab.resolve()).as_posix()
    except ValueError:
        return dst.resolve().as_posix()


def _write_names_yaml(dst: Path, names: tuple[str, ...], masks_dir: bool) -> None:
    lines = [
        f"path: {_yaml_path(dst)}",
        "train: images/train",
        "val: images/val",
    ]
    if masks_dir:
        lines.append("masks_dir: masks")
    lines.append("names:")
    lines.extend(f"  {i}: {name}" for i, name in enumerate(names))
    (dst / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _iter_rgb(rgb_root: Path, condition: str, split: str):
    split_dir = rgb_root / condition / split
    if not split_dir.is_dir():
        raise SystemExit(f"Missing images: {split_dir}")
    return sorted(split_dir.rglob("*_rgb_anon.png"))


def convert_semantic(rgb_root: Path, gt_zip: Path, dst: Path, conditions: list[str]) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    counts = {"train": 0, "val": 0}
    with zipfile.ZipFile(gt_zip) as z:
        names = set(z.namelist())
        for condition in conditions:
            for split in ("train", "val"):
                for img in _iter_rgb(rgb_root, condition, split):
                    stem = img.name[: -len("_rgb_anon.png")]
                    seq = img.parent.name
                    train_ids = f"gt/{condition}/{split}/{seq}/{stem}_gt_labelTrainIds.png"
                    inv_ids = f"gt/{condition}/{split}/{seq}/{stem}_gt_invIds.png"
                    if train_ids not in names or inv_ids not in names:
                        raise SystemExit(f"Missing mask for {img}")
                    mask = _read_zip_gray(z, train_ids)
                    inv = _read_zip_gray(z, inv_ids)
                    mask[inv == 1] = 255
                    out_stem = f"{condition}_{stem}" if len(conditions) > 1 else stem
                    _link_or_copy(img, dst / "images" / split / f"{out_stem}.png")
                    out_mask = dst / "masks" / split / f"{out_stem}.png"
                    out_mask.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(out_mask), mask):
                        raise SystemExit(f"Failed to write {out_mask}")
                    counts[split] += 1
    _write_names_yaml(dst, CITYSCAPES_19, masks_dir=True)
    print(f">> semantic train={counts['train']} val={counts['val']} -> {dst / 'data.yaml'}")


def _load_detection(det_zip: Path, condition: str, split: str) -> dict:
    member = f"gt_detection/{condition}/instancesonly_{condition}_{split}_gt_detection.json"
    with zipfile.ZipFile(det_zip) as z:
        return json.loads(z.read(member))


def convert_detect(rgb_root: Path, det_zip: Path, dst: Path, conditions: list[str]) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    box_counts = {"train": 0, "val": 0}
    image_counts = {"train": 0, "val": 0}
    for condition in conditions:
        for split in ("train", "val"):
            data = _load_detection(det_zip, condition, split)
            images = {img["id"]: img for img in data["images"]}
            by_image: dict[int, list[str]] = {img_id: [] for img_id in images}
            for ann in data["annotations"]:
                if ann.get("iscrowd", 0):
                    continue
                cls = DETECT_IDS.get(ann["category_id"])
                if cls is None:
                    raise SystemExit(f"Unexpected category_id {ann['category_id']}")
                image = images[ann["image_id"]]
                x, y, w, h = ann["bbox"]
                width = float(image["width"])
                height = float(image["height"])
                if w <= 1 or h <= 1:
                    continue
                cx = (x + w / 2) / width
                cy = (y + h / 2) / height
                nw = w / width
                nh = h / height
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < nw <= 1 and 0 < nh <= 1):
                    continue
                by_image[ann["image_id"]].append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                box_counts[split] += 1
            for image in images.values():
                file_name = image["file_name"].replace("\\", "/")
                src = rgb_root / file_name
                if not src.is_file():
                    raise SystemExit(f"Missing image {src}")
                stem = src.name[: -len("_rgb_anon.png")]
                out_stem = f"{condition}_{stem}" if len(conditions) > 1 else stem
                _link_or_copy(src, dst / "images" / split / f"{out_stem}.png")
                label = dst / "labels" / split / f"{out_stem}.txt"
                label.parent.mkdir(parents=True, exist_ok=True)
                label.write_text("\n".join(by_image[image["id"]]) + ("\n" if by_image[image["id"]] else ""), encoding="utf-8")
                image_counts[split] += 1
    _write_names_yaml(dst, DETECT_NAMES, masks_dir=False)
    print(
        f">> detect images train={image_counts['train']} val={image_counts['val']} "
        f"boxes train={box_counts['train']} val={box_counts['val']} -> {dst / 'data.yaml'}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert ACDC to YOLO semantic and detect layouts")
    p.add_argument("--rgb", type=Path, default=_DOWNLOADS / "rgb_anon_trainvaltest" / "rgb_anon")
    p.add_argument("--gt-zip", type=Path, default=_DOWNLOADS / "gt_trainval.zip")
    p.add_argument("--det-zip", type=Path, default=_DOWNLOADS / "gt_detection_trainval.zip")
    p.add_argument("--conditions", default="night", help="Comma-separated: fog,night,rain,snow")
    p.add_argument("--sem-dst", type=Path, default=_REPO / "yolo-lab" / "datasets" / "acdc-night-sem")
    p.add_argument("--det-dst", type=Path, default=_REPO / "yolo-lab" / "datasets" / "acdc-night-detect")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    conditions = [part.strip() for part in args.conditions.split(",") if part.strip()]
    if not args.rgb.is_dir():
        raise SystemExit(f"RGB root not found: {args.rgb}")
    if not args.gt_zip.is_file():
        raise SystemExit(f"Semantic zip not found: {args.gt_zip}")
    if not args.det_zip.is_file():
        raise SystemExit(f"Detection zip not found: {args.det_zip}")
    convert_semantic(args.rgb, args.gt_zip, args.sem_dst, conditions)
    convert_detect(args.rgb, args.det_zip, args.det_dst, conditions)


if __name__ == "__main__":
    main()
