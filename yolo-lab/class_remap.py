"""Align dataset class IDs to model.names by class name for fair YOLO val.

Roboflow exports remap COCO-subset labels to contiguous 0..nc-1 IDs. Pretrained
YOLO weights still emit COCO IDs (e.g. bottle=39). Ultralytics val uses
model.names for metrics, so those runs score near zero unless labels are
remapped. Fine-tuned weights already share the dataset name table — leave them
untouched.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import yaml


def normalize_names(names: dict | list) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(n) for i, n in enumerate(names)}


def load_data_yaml(data_yaml: Path) -> dict:
    with data_yaml.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_split_dir(data_yaml: Path, split_value: str) -> Path:
    """Resolve a yaml split path the same way Ultralytics check_det_dataset does."""
    raw = Path(split_value)
    if raw.is_absolute():
        return raw.resolve()
    root = data_yaml.parent.resolve()
    candidate = (root / raw).resolve()
    # Roboflow yamls often use '../valid/images' while the yaml already sits
    # inside the dataset root; Ultralytics strips the leading '../' fallback.
    if not candidate.exists() and split_value.startswith("../"):
        candidate = (root / split_value[3:]).resolve()
    return candidate


def names_aligned(model_names: dict[int, str], data_names: dict[int, str]) -> bool:
    return all(model_names.get(i) == name for i, name in data_names.items())


# Driving-dataset synonyms → COCO names. Unlisted names are dropped from pretrained val.
NAME_ALIASES = {
    "bike": "bicycle",
    "motor": "motorcycle",
}


def canonical_name(name: str) -> str:
    return NAME_ALIASES.get(name, name)


def build_dataset_to_model_map(
    model_names: dict[int, str],
    data_names: dict[int, str],
) -> dict[int, int]:
    name_to_model = {name: idx for idx, name in model_names.items()}
    mapping: dict[int, int] = {}
    skipped: list[str] = []
    for ds_i, name in data_names.items():
        target = canonical_name(name)
        if target in name_to_model:
            mapping[ds_i] = name_to_model[target]
        else:
            skipped.append(f"{ds_i}:{name}")
    if skipped:
        print(">> Classes with no COCO name (boxes dropped in pretrained val): " + ", ".join(skipped))
    if not mapping:
        raise SystemExit("No dataset class names overlap with model.names; cannot remap")
    return mapping


def _link_or_copy_file(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _remap_label_file(src: Path, dst: Path, mapping: dict[int, int]) -> tuple[int, int]:
    """Rewrite one YOLO label file. Returns (kept_boxes, dropped_boxes)."""
    kept = 0
    dropped = 0
    lines_out: list[str] = []
    text = src.read_text(encoding="utf-8").strip()
    if text:
        for line in text.splitlines():
            parts = line.split()
            if not parts:
                continue
            old_id = int(float(parts[0]))
            if old_id not in mapping:
                dropped += 1
                continue
            parts[0] = str(mapping[old_id])
            lines_out.append(" ".join(parts))
            kept += 1
    dst.write_text(("\n".join(lines_out) + ("\n" if lines_out else "")), encoding="utf-8")
    return kept, dropped


def prepare_remapped_val_data(
    data_yaml: Path,
    model_names: dict[int, str],
    cache_root: Path,
) -> tuple[Path, dict[int, int]]:
    """Build a val-only dataset whose label IDs match model.names.

    Returns (path_to_temp_data_yaml, dataset_id_to_model_id).
    """
    cfg = load_data_yaml(data_yaml)
    data_names = normalize_names(cfg["names"])
    mapping = build_dataset_to_model_map(model_names, data_names)

    images_dir = resolve_split_dir(data_yaml, cfg["val"])
    labels_dir = Path(str(images_dir).replace(f"{os.sep}images", f"{os.sep}labels", 1))
    if not labels_dir.is_dir():
        # fallback: sibling labels/
        labels_dir = images_dir.parent / "labels"
    if not images_dir.is_dir():
        raise SystemExit(f"Val images not found: {images_dir}")
    if not labels_dir.is_dir():
        raise SystemExit(f"Val labels not found: {labels_dir}")

    dataset_key = data_yaml.parent.resolve().name
    out_root = cache_root / dataset_key
    out_images = out_root / "images"
    out_labels = out_root / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not image_files:
        raise SystemExit(f"No images under {images_dir}")

    kept_total = 0
    dropped_total = 0
    for img in image_files:
        _link_or_copy_file(img, out_images / img.name)
        src_lbl = labels_dir / f"{img.stem}.txt"
        dst_lbl = out_labels / f"{img.stem}.txt"
        if src_lbl.is_file():
            kept, dropped = _remap_label_file(src_lbl, dst_lbl, mapping)
            kept_total += kept
            dropped_total += dropped
        else:
            dst_lbl.write_text("", encoding="utf-8")

    # Drop stale Ultralytics caches so remapped IDs are re-scanned
    for cache in out_root.glob("*.cache"):
        cache.unlink(missing_ok=True)

    out_yaml = out_root / "data.yaml"
    names_block = {int(i): n for i, n in sorted(model_names.items())}
    payload = {
        "path": str(out_root.resolve()).replace("\\", "/"),
        "train": "images",  # unused for val; required by some loaders
        "val": "images",
        "nc": len(names_block),
        "names": names_block,
    }
    with out_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

    print(
        f">> Class remap: {data_yaml} → model.names "
        f"({len(mapping)} classes, {kept_total} boxes kept, {dropped_total} dropped)"
    )
    print(f">> Remap dataset: {out_yaml}")
    for ds_i, model_i in sorted(mapping.items()):
        print(f"   {ds_i} ({data_names[ds_i]}) → {model_i} ({model_names[model_i]})")

    return out_yaml, mapping


def resolve_val_data(
    data_yaml: str | Path,
    model_names: dict[int, str],
    cache_root: Path,
) -> tuple[str, bool, dict[int, int] | None]:
    """Return (data_path_for_val, remapped, mapping_or_none)."""
    data_path = Path(data_yaml)
    if not data_path.is_file():
        raise SystemExit(f"data yaml not found: {data_path}")

    cfg = load_data_yaml(data_path)
    data_names = normalize_names(cfg["names"])
    model_names = normalize_names(model_names)

    if names_aligned(model_names, data_names):
        print(">> Class IDs already aligned with model.names; no remap")
        return str(data_path), False, None

    out_yaml, mapping = prepare_remapped_val_data(data_path, model_names, cache_root)
    return str(out_yaml), True, mapping
