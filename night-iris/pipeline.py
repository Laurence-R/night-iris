"""Run Night-Iris stages on one image or a folder."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import cv2
import torch

from enhance import EnhanceEngine
from fuse import fuse
from sem_object import DEFAULT_OBJECT_CLASSES, ObjectSemEngine
from sem_shadow import ShadowSemEngine, resolve_shadow_model
from timing import StageTiming, cpu_ms
from visualize import dump_stages

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise SystemExit(f"Input dir not found: {folder}")
    files = [
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    files.sort(key=lambda p: p.name)
    return files


def pick_device() -> str:
    return "0" if torch.cuda.is_available() else "cpu"


class Pipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = pick_device()
        object_cfg = cfg["sem_object"]
        shadow_cfg = cfg["sem_shadow"]
        clahe_cfg = cfg["clahe"]
        shadow_path = resolve_shadow_model(shadow_cfg)
        self.object_sem = ObjectSemEngine(
            model=str(object_cfg["model"]),
            imgsz=int(object_cfg.get("imgsz", 640)),
            object_classes=list(object_cfg.get("object_classes") or DEFAULT_OBJECT_CLASSES),
            device=self.device,
        )
        grid = tuple(int(x) for x in clahe_cfg.get("grid", [8, 8]))
        clahe_device = "cuda" if self.device != "cpu" else "cpu"
        self.enhance = EnhanceEngine(
            grid=grid,
            clip_limit=float(clahe_cfg.get("clip_limit", 4.0)),
            device=clahe_device,
        )
        ids = list(shadow_cfg.get("dark_class_ids") or [])
        self.shadow_sem = ShadowSemEngine(
            str(shadow_path),
            ids,
            self.device,
            imgsz=int(shadow_cfg.get("imgsz", 640)),
        )

    def process_one(
        self,
        image_path: Path,
        out_root: Path,
        *,
        warmup: bool = False,
        dump: bool = False,
    ) -> tuple[Path, StageTiming]:
        t0 = perf_counter()
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        imread_ms = (perf_counter() - t0) * 1000.0
        if bgr is None:
            raise SystemExit(f"Failed to read image: {image_path}")
        rgb, color_ms = cpu_ms(cv2.cvtColor, bgr, cv2.COLOR_BGR2RGB)

        viz = self.cfg.get("viz", {})
        min_region_area = int(viz.get("min_region_area", 256))
        objects = self.object_sem.predict(bgr, dump=dump, min_region_area=min_region_area)
        dark = self.shadow_sem.predict(bgr, objects.object_mask)
        clahe_rgb, clahe_ms = self.enhance.clahe_rgb(rgb)

        output_rgb, fuse_ms = cpu_ms(fuse, rgb, clahe_rgb, dark.dark_mask)

        output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
        ldr_dir = out_root / "ldr"
        ldr_dir.mkdir(parents=True, exist_ok=True)
        ldr_path = ldr_dir / image_path.name
        cv2.imwrite(str(ldr_path), output_bgr)

        viz_ms = 0.0
        dest = ldr_path
        if dump:
            t_viz = perf_counter()
            debug_dir = out_root / "debug" / image_path.stem
            dump_stages(
                debug_dir,
                orig_bgr=bgr,
                objects=objects,
                dark=dark,
                clahe_full_bgr=cv2.cvtColor(clahe_rgb, cv2.COLOR_RGB2BGR),
                output_bgr=output_bgr,
                min_region_area=min_region_area,
                save_crops=bool(viz.get("save_crops", True)),
            )
            viz_ms = (perf_counter() - t_viz) * 1000.0 + float(objects.plot_ms)
            dest = debug_dir

        timing = StageTiming(
            image=image_path.name,
            warmup=warmup,
            color_ms=color_ms,
            seg_infer_ms=objects.infer_ms,
            seg_mask_ms=objects.mask_ms,
            sem_infer_ms=dark.infer_ms,
            sem_mask_ms=dark.mask_ms,
            clahe_ms=clahe_ms,
            fuse_ms=fuse_ms,
            imread_ms=imread_ms,
            viz_ms=viz_ms,
            seg_pre_ms=objects.pre_ms,
            seg_post_ms=objects.post_ms,
            sem_pre_ms=dark.pre_ms,
            sem_post_ms=dark.post_ms,
        )
        return dest, timing
