import argparse
import random
import tomllib
from pathlib import Path

from pipeline import Pipeline, list_images
from timing import summarize, write_timings_charts


def load_config(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def write_run_config(dest: Path, cfg: dict, n_images: int) -> None:
    paths = cfg["paths"]
    object_cfg = cfg["sem_object"]
    shadow_cfg = cfg["sem_shadow"]
    clahe = cfg["clahe"]
    viz = cfg.get("viz", {})
    timing = cfg.get("timing", {})
    grid = clahe.get("grid", [8, 8])
    dark_ids = shadow_cfg.get("dark_class_ids") or []
    ids_txt = ", ".join(str(int(x)) for x in dark_ids)
    object_classes = list(object_cfg.get("object_classes") or [])
    cls_txt = ", ".join(f'"{c}"' for c in object_classes)
    text = (
        f'[paths]\n'
        f'input_dir = "{paths["input_dir"]}"\n'
        f'output_dir = "{paths["output_dir"]}"\n'
        f"\n"
        f"[sem_object]\n"
        f'model = "{object_cfg["model"]}"\n'
        f'imgsz = {int(object_cfg.get("imgsz", 640))}\n'
        f"object_classes = [{cls_txt}]\n"
        f"\n"
        f"[sem_shadow]\n"
        f'model = "{shadow_cfg["model"]}"\n'
        f'imgsz = {int(shadow_cfg.get("imgsz", 640))}\n'
        f"dark_class_ids = [{ids_txt}]\n"
        f"\n"
        f"[clahe]\n"
        f"grid = [{int(grid[0])}, {int(grid[1])}]\n"
        f"clip_limit = {float(clahe.get('clip_limit', 4.0))}\n"
        f"\n"
        f"[viz]\n"
        f"min_region_area = {int(viz.get('min_region_area', 256))}\n"
        f'save_crops = {"true" if viz.get("save_crops", True) else "false"}\n'
        f"sample = {int(viz.get('sample', 8))}\n"
        f"sample_seed = {int(viz.get('sample_seed', 0))}\n"
        f"\n"
        f"[timing]\n"
        f"warmup = {int(timing.get('warmup', 1))}\n"
        f"\n"
        f"[run]\n"
        f"n_images = {n_images}\n"
    )
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "run_config.toml"
    out.write_text(text, encoding="utf-8")
    print(f">> Run config saved to: {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Night-Iris LDR preprocessor")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--input", default=None, help="Override input directory")
    parser.add_argument("--output", default=None, help="Override output directory")
    parser.add_argument("--count", type=int, default=None, help="Limit number of images")
    return parser.parse_args()


def pick_sample_paths(items: list[Path], sample: int, seed: int) -> set[Path]:
    if sample <= 0 or sample >= len(items):
        return set(items)
    return set(random.Random(seed).sample(items, sample))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.input is not None:
        cfg["paths"]["input_dir"] = args.input
    if args.output is not None:
        cfg["paths"]["output_dir"] = args.output

    input_dir = Path(cfg["paths"]["input_dir"])
    output_dir = Path(cfg["paths"]["output_dir"])
    items = list_images(input_dir)
    if args.count is not None:
        items = items[: args.count]
    if not items:
        raise SystemExit(f"No images under {input_dir}")

    warmup_n = int(cfg.get("timing", {}).get("warmup", 1))
    viz = cfg.get("viz", {})
    sample_n = int(viz.get("sample", 8))
    sample_seed = int(viz.get("sample_seed", 0))
    dump_set = pick_sample_paths(items, sample_n, sample_seed)
    print(
        f">> Loading pipeline (sem_object={cfg['sem_object']['model']}  "
        f"sem_shadow={cfg['sem_shadow']['model']})"
    )
    pipe = Pipeline(cfg)
    print(
        f">> device={pipe.device}  n={len(items)}  warmup={warmup_n}  "
        f"dump={len(dump_set)}/{len(items)}"
    )
    write_run_config(output_dir, cfg, len(items))

    rows = []
    for i, path in enumerate(items, 1):
        warmup = i <= warmup_n
        dump = path in dump_set
        dest, timing = pipe.process_one(path, output_dir, warmup=warmup, dump=dump)
        rows.append(timing)
        tag = " warmup" if warmup else ""
        dump_tag = " dump" if dump else ""
        print(
            f"[{i}/{len(items)}] {path.name}{tag}{dump_tag}  algo={timing.algo_ms:.1f}ms"
            f"  seg={timing.seg_infer_ms:.1f}  sem={timing.sem_infer_ms:.1f}"
            f"  clahe={timing.clahe_ms:.1f}  fuse={timing.fuse_ms:.1f}"
            f"  -> {dest}"
        )

    stale_csv = output_dir / "timings.csv"
    if stale_csv.exists():
        stale_csv.unlink()
    chart_path = write_timings_charts(output_dir / "timings.png", rows)
    print(f">> Timings chart: {chart_path.resolve()}")
    summarize(
        rows,
        (
            "algo_ms",
            "color_ms",
            "seg_infer_ms",
            "seg_mask_ms",
            "sem_infer_ms",
            "sem_mask_ms",
            "clahe_ms",
            "fuse_ms",
        ),
    )
    summarize(
        rows,
        (
            "imread_ms",
            "viz_ms",
            "seg_pre_ms",
            "seg_post_ms",
            "sem_pre_ms",
            "sem_post_ms",
        ),
        title="Info (not in algo_ms): I/O, viz, YOLO pre/post",
    )
    print(f"Done. LDR in {(output_dir / 'ldr').resolve()}  debug dumps in {(output_dir / 'debug').resolve()}")


if __name__ == "__main__":
    main()
