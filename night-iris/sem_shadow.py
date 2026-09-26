"""YOLO-sem bright/dark → dark_mask. Protected objects are logical ignore."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from ultralytics import YOLO

from timing import yolo_speed_ms


@dataclass(frozen=True)
class DarkMap:
    dark_mask: np.ndarray  # bool (H, W), already restricted to ~object_mask
    n_dark: int
    infer_ms: float = 0.0
    mask_ms: float = 0.0
    pre_ms: float = 0.0
    post_ms: float = 0.0


def resolve_shadow_model(shadow_cfg: dict) -> Path:
    model = str(shadow_cfg.get("model", "") or "").strip()
    if not model:
        raise SystemExit("sem_shadow.model is required (path to YOLO-sem bright/dark weights)")
    path = Path(model)
    if not path.exists():
        raise SystemExit(f"sem_shadow.model not found: {model}")
    return path


class ShadowSemEngine:
    def __init__(self, model: str, dark_class_ids: list[int], device: str, imgsz: int = 640):
        self.model = YOLO(model)
        self.dark_class_ids = [int(x) for x in dark_class_ids]
        self.device = device
        self.imgsz = int(imgsz)

    def _resolve_dark_ids(self, names: dict | list) -> set[int]:
        if self.dark_class_ids:
            return set(self.dark_class_ids)
        if isinstance(names, dict):
            items = names.items()
        else:
            items = enumerate(names)
        found = [int(i) for i, n in items if str(n).lower() == "dark"]
        if not found:
            raise SystemExit(
                "sem_shadow.dark_class_ids is empty and no class named 'dark' in the YOLO-sem model"
            )
        return set(found)

    def predict(self, bgr: np.ndarray, object_mask: np.ndarray) -> DarkMap:
        h, w = bgr.shape[:2]
        results = self.model.predict(bgr, verbose=False, device=self.device, imgsz=self.imgsz)
        r = results[0]
        pre_ms, infer_ms, post_ms = yolo_speed_ms(r)
        semantic = getattr(r, "semantic_mask", None)
        if semantic is None:
            raise SystemExit("YOLO-sem result has no semantic_mask; check the checkpoint task")
        data = semantic.data
        if isinstance(data, torch.Tensor):
            class_map = data.detach().cpu().numpy()
        else:
            class_map = np.asarray(data)
        t_mask = perf_counter()
        if class_map.shape[:2] != (h, w):
            import cv2

            class_map = cv2.resize(
                class_map.astype(np.float32),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)
        dark_ids = self._resolve_dark_ids(r.names)
        dark = np.isin(class_map, list(dark_ids)) & (~object_mask)
        mask_ms = (perf_counter() - t_mask) * 1000.0
        return DarkMap(
            dark_mask=dark,
            n_dark=int(dark.sum()),
            infer_ms=infer_ms,
            mask_ms=mask_ms,
            pre_ms=pre_ms,
            post_ms=post_ms,
        )
