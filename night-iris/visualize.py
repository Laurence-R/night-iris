"""Stage dumps for seg / sem / enhance / output."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from sem_object import ObjectSemResult, source_1_display
from sem_shadow import DarkMap


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _banner(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), thickness=-1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _with_legend(overlay: np.ndarray, items: list[tuple[str, tuple[int, int, int], bool]]) -> np.ndarray:
    """Append a right-side legend for classes present in this frame."""
    if overlay.size == 0:
        return overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    row_h = 22
    pad = 8
    swatch = 14
    title = "in frame  (* = protect)"
    texts = [f"{'* ' if protected else '  '}{name}" for name, _color, protected in items]
    widths = [cv2.getTextSize(title, font, scale, 1)[0][0]]
    widths.extend(cv2.getTextSize(text, font, scale, 1)[0][0] for text in texts)
    panel_w = pad * 3 + swatch + (max(widths) if widths else 80)
    panel = np.full((overlay.shape[0], panel_w, 3), 24, dtype=np.uint8)
    y = pad + 16
    cv2.putText(panel, title, (pad, y), font, scale, (210, 210, 210), 1, cv2.LINE_AA)
    y += row_h
    for (_name, color, protected), text in zip(items, texts):
        if y + 6 > overlay.shape[0] - pad:
            break
        x0, y0 = pad, y - 12
        cv2.rectangle(panel, (x0, y0), (x0 + swatch, y0 + swatch), color, thickness=-1)
        border = (255, 255, 255) if protected else (90, 90, 90)
        cv2.rectangle(panel, (x0, y0), (x0 + swatch, y0 + swatch), border, thickness=1)
        cv2.putText(panel, text, (pad * 2 + swatch, y), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
        y += row_h
    return np.concatenate([overlay, panel], axis=1)


def overlay_sem_bgr(source_1_bgr: np.ndarray, dark: DarkMap, object_mask: np.ndarray) -> np.ndarray:
    vis = source_1_bgr.astype(np.float32)
    bright = (~object_mask) & (~dark.dark_mask)
    vis[dark.dark_mask] = vis[dark.dark_mask] * 0.42 + np.array([40.0, 40.0, 210.0]) * 0.58
    vis[bright] = vis[bright] * 0.42 + np.array([210.0, 90.0, 40.0]) * 0.58
    label = f"yolo-sem  dark_px={dark.n_dark}"
    return _banner(vis.clip(0, 255).astype(np.uint8), label)


def _region_crops(
    orig_bgr: np.ndarray,
    enhanced_bgr: np.ndarray,
    dark_mask: np.ndarray,
    min_area: int,
) -> list[np.ndarray]:
    mask_u8 = dark_mask.astype(np.uint8)
    num, _labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    h, w = dark_mask.shape
    pages: list[np.ndarray] = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        pad = 2
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
        left = orig_bgr[y1:y2, x1:x2]
        right = enhanced_bgr[y1:y2, x1:x2]
        gap = np.full((left.shape[0], 4, 3), 32, dtype=np.uint8)
        pair = np.concatenate([left, gap, right], axis=1)
        pair = _banner(pair, f"region {len(pages):03d}  area={area}  orig | fused")
        pages.append(pair)
    return pages


def dump_stages(
    out_dir: Path,
    *,
    orig_bgr: np.ndarray,
    objects: ObjectSemResult,
    dark: DarkMap,
    clahe_full_bgr: np.ndarray,
    output_bgr: np.ndarray,
    min_region_area: int,
    save_crops: bool,
) -> None:
    _ensure_dir(out_dir)
    src1 = source_1_display(orig_bgr, objects.object_mask)
    n_protect = sum(1 for _name, _color, protected in objects.legend_items if protected)
    overlay = _with_legend(objects.overlay_bgr, objects.legend_items)
    cv2.imwrite(
        str(out_dir / "01_seg.png"),
        _banner(
            overlay,
            f"sem-object  protect_px={objects.n_protect_px}  "
            f"protect_cls={n_protect}/{len(objects.legend_items)}",
        ),
    )
    cv2.imwrite(str(out_dir / "01_source_1.png"), _banner(src1, "source_1  objects=ignore"))
    if save_crops:
        crop_dir = _ensure_dir(out_dir / "01_seg_crops")
        for name, crop in objects.crops:
            cv2.imwrite(str(crop_dir / f"{name}.png"), crop)

    cv2.imwrite(str(out_dir / "02_sem.png"), overlay_sem_bgr(src1, dark, objects.object_mask))
    cv2.imwrite(str(out_dir / "03_clahe_full.png"), _banner(clahe_full_bgr, "full-frame CLAHE  before write-back"))
    if save_crops:
        region_dir = _ensure_dir(out_dir / "03_enhance_regions")
        for i, page in enumerate(_region_crops(orig_bgr, output_bgr, dark.dark_mask, min_region_area)):
            cv2.imwrite(str(region_dir / f"{i:03d}.png"), page)
    cv2.imwrite(str(out_dir / "04_output.png"), _banner(output_bgr, "fusion  dark=CLAHE  else=original"))
