"""YOLO-sem Cityscapes parse → object_mask for protected thing classes."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from timing import yolo_speed_ms

DEFAULT_OBJECT_CLASSES = (
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
)


@dataclass(frozen=True)
class ObjectSemResult:
    object_mask: np.ndarray  # bool (H, W)
    overlay_bgr: np.ndarray
    class_map: np.ndarray  # int (H, W)
    names: dict[int, str]
    protect_ids: frozenset[int]
    legend_items: list[tuple[str, tuple[int, int, int], bool]] = field(default_factory=list)
    crops: list[tuple[str, np.ndarray]] = field(default_factory=list)
    n_protect_px: int = 0
    infer_ms: float = 0.0
    mask_ms: float = 0.0
    pre_ms: float = 0.0
    post_ms: float = 0.0
    plot_ms: float = 0.0


def _normalize_name(name: str) -> str:
    return " ".join(str(name).lower().replace("_", " ").replace("-", " ").split())


def names_to_dict(names: dict | list) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(n) for i, n in enumerate(names)}


def resolve_object_ids(names: dict | list, wanted: list[str]) -> frozenset[int]:
    by_norm = {_normalize_name(v): i for i, v in names_to_dict(names).items()}
    found: list[int] = []
    missing: list[str] = []
    for raw in wanted:
        key = _normalize_name(raw)
        if key not in by_norm:
            missing.append(raw)
            continue
        found.append(by_norm[key])
    if missing:
        available = ", ".join(names_to_dict(names).values())
        raise SystemExit(f"object_classes not in model.names: {missing}. Available: {available}")
    return frozenset(found)


def _semantic_class_map(result, h: int, w: int) -> np.ndarray:
    semantic = getattr(result, "semantic_mask", None)
    if semantic is None:
        raise SystemExit("YOLO-sem result has no semantic_mask; first-stage weights must be task=semantic")
    data = semantic.data
    if isinstance(data, torch.Tensor):
        class_map = data.detach().cpu().numpy()
    else:
        class_map = np.asarray(data)
    class_map = np.squeeze(class_map)
    if class_map.ndim != 2:
        raise SystemExit(f"Unexpected semantic_mask shape: {class_map.shape}")
    if class_map.shape != (h, w):
        class_map = cv2.resize(
            class_map.astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )
    return class_map.astype(np.int32)


def _legend_items(
    overlay: np.ndarray,
    class_map: np.ndarray,
    names: dict[int, str],
    protect_ids: frozenset[int],
) -> list[tuple[str, tuple[int, int, int], bool]]:
    items: list[tuple[str, tuple[int, int, int], bool]] = []
    for cid in sorted(int(v) for v in np.unique(class_map)):
        ys, xs = np.nonzero(class_map == cid)
        if ys.size == 0:
            continue
        pick = np.linspace(0, ys.size - 1, num=min(200, ys.size), dtype=int)
        color = tuple(int(v) for v in np.median(overlay[ys[pick], xs[pick]], axis=0))
        items.append((names.get(cid, str(cid)), color, cid in protect_ids))
    return items


def _object_crops(
    bgr: np.ndarray,
    object_mask: np.ndarray,
    class_map: np.ndarray,
    names: dict[int, str],
    min_area: int,
) -> list[tuple[str, np.ndarray]]:
    mask_u8 = object_mask.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    h, w = object_mask.shape
    crops: list[tuple[str, np.ndarray]] = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + bw), min(h, y + bh)
        if x2 <= x1 or y2 <= y1:
            continue
        region = class_map[labels == i]
        if region.size == 0:
            continue
        cid = int(np.bincount(region.ravel()).argmax())
        label = names.get(cid, str(cid)).replace(" ", "_")
        crops.append((f"{len(crops):03d}_{label}", bgr[y1:y2, x1:x2].copy()))
    return crops


class ObjectSemEngine:
    def __init__(self, model: str, imgsz: int, object_classes: list[str], device: str):
        self.model = YOLO(model)
        self.imgsz = imgsz
        self.object_classes = [str(x) for x in object_classes] or list(DEFAULT_OBJECT_CLASSES)
        self.device = device
        self.protect_ids = resolve_object_ids(self.model.names, self.object_classes)

    def predict(self, bgr: np.ndarray, *, dump: bool = False, min_region_area: int = 256) -> ObjectSemResult:
        h, w = bgr.shape[:2]
        results = self.model.predict(
            bgr,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
        )
        r = results[0]
        pre_ms, infer_ms, post_ms = yolo_speed_ms(r)
        names = names_to_dict(r.names)
        t_mask = perf_counter()
        class_map = _semantic_class_map(r, h, w)
        object_mask = np.isin(class_map, list(self.protect_ids))
        mask_ms = (perf_counter() - t_mask) * 1000.0

        overlay = np.zeros((0, 0, 3), dtype=np.uint8)
        legend_items: list[tuple[str, tuple[int, int, int], bool]] = []
        crops: list[tuple[str, np.ndarray]] = []
        plot_ms = 0.0
        if dump:
            t_plot = perf_counter()
            overlay = r.plot()
            if overlay.shape[:2] != (h, w):
                overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_NEAREST)
            legend_items = _legend_items(overlay, class_map, names, self.protect_ids)
            crops = _object_crops(bgr, object_mask, class_map, names, min_region_area)
            plot_ms = (perf_counter() - t_plot) * 1000.0

        return ObjectSemResult(
            object_mask=object_mask,
            overlay_bgr=overlay,
            class_map=class_map,
            names=names,
            protect_ids=self.protect_ids,
            legend_items=legend_items,
            crops=crops,
            n_protect_px=int(object_mask.sum()),
            infer_ms=infer_ms,
            mask_ms=mask_ms,
            pre_ms=pre_ms,
            post_ms=post_ms,
            plot_ms=plot_ms,
        )


def source_1_display(bgr: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
    """Visualization only: dim object pixels. Not used as filled sem input."""
    out = bgr.copy()
    if object_mask.any():
        dim = (out[object_mask].astype(np.float32) * 0.22 + 36.0).clip(0, 255).astype(np.uint8)
        out[object_mask] = dim
    return out
