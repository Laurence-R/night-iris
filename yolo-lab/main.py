import argparse
import json
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import torch
from ultralytics import YOLO

from class_remap import resolve_val_data

MODEL_SIZES = ("n", "s", "m", "l", "x")
RUNS_PROJECT = str(Path("runs").resolve())
REMAP_CACHE = Path("experiments") / "_class_remap"


def load_config(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def resolve_model_paths(cfg: dict, model_size: str, model_override: str | None) -> list[tuple[str, str]]:
    """Return list of (size_label, model_path)."""
    if model_override:
        return [(model_size if model_size != "all" else "custom", model_override)]

    template = cfg.get("model_template")
    if model_size == "all":
        if not template:
            raise SystemExit("--model-size all requires model_template in the config")
        return [(size, template.format(size=size)) for size in MODEL_SIZES]

    if template:
        return [(model_size, template.format(size=model_size))]
    return [(model_size, cfg["model"])]


def experiment_dir(name: str) -> Path:
    path = Path("experiments") / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def cfg_task(cfg: dict) -> str:
    return str(cfg.get("task", "detect")).strip().lower() or "detect"


def write_snapshot(exp_dir: Path, cfg: dict, model_path: str) -> None:
    train = cfg.get("train", {})
    val = cfg.get("val", {})
    predict = cfg.get("predict", {})
    export = cfg.get("export", {})
    quantize = export.get("quantize", 16)

    text = (
        f'name = "{cfg["name"]}"\n'
        f'mode = "{cfg["mode"]}"\n'
        f'model = "{model_path}"\n'
        f'data = "{cfg.get("data", "")}"\n'
        f'device = {cfg.get("device", 0)}\n'
        f'\n'
        f'[train]\n'
        f'epochs = {train.get("epochs", 50)}\n'
        f'patience = {train.get("patience", 15)}\n'
        f'batch = {train.get("batch", 8)}\n'
        f'imgsz = {train.get("imgsz", 640)}\n'
        f'\n'
        f'[val]\n'
        f'batch = {val.get("batch", 1)}\n'
        f'imgsz = {val.get("imgsz", 640)}\n'
        f'\n'
        f'[predict]\n'
        f'source = "{predict.get("source", "datasets/bdd10k-night/70-15-15/valid/images")}"\n'
        f'conf = {predict.get("conf", 0.25)}\n'
        f'batch = {predict.get("batch", 1)}\n'
        f'\n'
        f'[export]\n'
        f'format = "{export.get("format", "engine")}"\n'
        f'quantize = {quantize}\n'
        f'batch = {export.get("batch", 1)}\n'
        f'imgsz = {export.get("imgsz", 640)}\n'
    )
    (exp_dir / "config.snapshot.toml").write_text(text, encoding="utf-8")


def train_model(model: YOLO, cfg: dict) -> None:
    data = Path(str(cfg["data"]))
    if not data.is_file():
        hint = ""
        if cfg_task(cfg) == "semantic":
            hint = "\nConvert first: uv run --package dataset-transform --directory dataset-transform python trans_script/sbu-shadow.py"
        raise SystemExit(f"Dataset yaml not found: {data}{hint}")
    t = cfg.get("train", {})
    results = model.train(
        data=str(data),
        epochs=t.get("epochs", 50),
        patience=t.get("patience", 15),
        batch=t.get("batch", 8),
        imgsz=t.get("imgsz", 640),
        device=cfg.get("device", 0),
        project=RUNS_PROJECT,
        name=cfg["name"],
    )
    save_dir = getattr(results, "save_dir", None)
    if save_dir is not None:
        weights = Path(save_dir) / "weights" / "best.pt"
        print(f">> best weights: {weights}")
        if cfg_task(cfg) == "semantic":
            print(f">> Set [sem_shadow].model in night-iris/configs/default.toml to: {weights.as_posix()}")


def predict_model(model: YOLO, cfg: dict) -> None:
    p = cfg.get("predict", {})
    model.predict(
        source=p.get("source", "datasets/bdd10k-night/70-15-15/valid/images"),
        conf=p.get("conf", 0.25),
        batch=p.get("batch", 1),
        save=True,
        device=cfg.get("device", 0),
        project=RUNS_PROJECT,
    )


def export_model(model: YOLO, cfg: dict) -> None:
    e = cfg.get("export", {})
    # Ultralytics 8.4+: use quantize=16 for FP16 (half= is deprecated)
    quantize = e.get("quantize", 16)
    batch = int(e.get("batch", 1))
    imgsz = int(e.get("imgsz", 640))
    print(f"Exporting TensorRT engine: quantize={quantize} batch={batch} imgsz={imgsz}")
    model.export(
        format=e.get("format", "engine"),
        quantize=quantize,
        batch=batch,
        imgsz=imgsz,
        device=cfg.get("device", 0),
    )


def _semantic_miou(results) -> float | None:
    for obj in (results, getattr(results, "seg", None), getattr(results, "semantic", None)):
        if obj is None:
            continue
        for attr in ("miou", "mIoU"):
            if hasattr(obj, attr):
                val = getattr(obj, attr)
                if val is not None:
                    return float(val)
    metrics = getattr(results, "results_dict", None)
    if isinstance(metrics, dict):
        for key in ("metrics/mIoU", "mIoU", "miou"):
            if key in metrics:
                return float(metrics[key])
    return None


def validate_model(model: YOLO, cfg: dict, model_path: str, size_label: str = "n") -> dict:
    v = cfg.get("val", {})
    batch = int(v.get("batch", 1))
    imgsz = int(v.get("imgsz", 640))
    task = cfg_task(cfg)

    remapped = False
    mapping = None
    val_data = cfg["data"]
    val_kwargs: dict = {
        "data": val_data,
        "device": cfg.get("device", 0),
        "batch": batch,
        "imgsz": imgsz,
        "project": RUNS_PROJECT,
        "plots": False,  # metrics-only; avoids 20× plot I/O on full ablations
    }
    # Detect labels use contiguous Roboflow IDs; COCO pretrained heads do not.
    # Semantic masks are already class indices — never run box class_remap.
    if task != "semantic":
        REMAP_CACHE.mkdir(parents=True, exist_ok=True)
        val_data, remapped, mapping = resolve_val_data(cfg["data"], model.names, REMAP_CACHE)
        val_kwargs["data"] = val_data
        if remapped and mapping:
            val_kwargs["classes"] = sorted(set(mapping.values()))

    print(f"Validating with batch={batch} imgsz={imgsz} remapped={remapped} task={task}")
    results = model.val(**val_kwargs)

    preprocess_time = results.speed["preprocess"]
    inference_time = results.speed["inference"]
    postprocess_time = results.speed["postprocess"]
    total_time_per_image = preprocess_time + inference_time + postprocess_time
    fps = 1000 / total_time_per_image if total_time_per_image > 0 else 0.0

    map50 = None
    map50_95 = None
    miou = None
    print("--- 驗證結果與平均單張耗時 ---")
    if task == "semantic":
        miou = _semantic_miou(results)
        if miou is not None:
            print(f"精確度 mIoU: {miou:.4f}")
    else:
        map50 = float(results.box.map50) if hasattr(results.box, "map50") else None
        map50_95 = float(results.box.map)
        if map50 is not None:
            print(f"精確度 mAP50: {map50:.4f}")
        print(f"精確度 mAP50-95: {map50_95:.4f}")
    print(f"batch: {batch}")
    print(f"class_remap: {remapped}")
    print(f"平均前處理: {preprocess_time:.2f} ms / 張")
    print(f"平均模型推論: {inference_time:.2f} ms / 張")
    print(f"平均後處理: {postprocess_time:.2f} ms / 張")
    print(f"單張總耗時: {total_time_per_image:.2f} ms")
    print(f"等效吞吐量 (FPS): {fps:.2f}")

    metrics = {
        "name": cfg["name"],
        "mode": "val",
        "model_size": size_label,
        "model": model_path,
        "data": cfg["data"],
        "data_effective": val_data,
        "class_remap": remapped,
        "class_remap_map": {str(k): v for k, v in mapping.items()} if mapping else None,
        "device": cfg.get("device", 0),
        "batch": batch,
        "imgsz": imgsz,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "map50": map50,
        "map50_95": map50_95,
        "miou": miou,
        "preprocess_ms": preprocess_time,
        "inference_ms": inference_time,
        "postprocess_ms": postprocess_time,
        "total_ms": total_time_per_image,
        "fps": fps,
    }

    exp_dir = experiment_dir(cfg["name"])
    write_snapshot(exp_dir, cfg, model_path)
    payload = json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    metrics_size_path = exp_dir / f"metrics_{size_label}.json"
    metrics_size_path.write_text(payload, encoding="utf-8")
    # Keep metrics.json as the latest run for backward compatibility
    (exp_dir / "metrics.json").write_text(payload, encoding="utf-8")
    print(f">> Metrics saved to: {metrics_size_path}")
    return metrics


def run_one(cfg: dict, model_path: str, size_label: str) -> None:
    if not Path(model_path).exists():
        raise SystemExit(f"Model not found: {model_path}")

    print(f"\n=== Experiment {cfg['name']} | size={size_label} | mode={cfg['mode']} | model={model_path} ===")
    # TensorRT engines do not always infer task from the file; force detect.
    yolo_kwargs = {"task": "detect"} if Path(model_path).suffix.lower() == ".engine" else {}
    model = YOLO(model_path, **yolo_kwargs)
    mode = cfg["mode"]

    if mode == "train":
        exp_dir = experiment_dir(cfg["name"])
        write_snapshot(exp_dir, cfg, model_path)
        train_model(model, cfg)
    elif mode == "val":
        validate_model(model, cfg, model_path, size_label=size_label)
    elif mode == "predict":
        exp_dir = experiment_dir(cfg["name"])
        write_snapshot(exp_dir, cfg, model_path)
        predict_model(model, cfg)
    elif mode == "export":
        exp_dir = experiment_dir(cfg["name"])
        write_snapshot(exp_dir, cfg, model_path)
        export_model(model, cfg)
    else:
        raise SystemExit(f"Unknown mode: {mode!r} (expected train|val|predict|export)")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO26 ablation experiment runner")
    parser.add_argument(
        "--config",
        default="configs/detect_bdd_orig_val_pretrained.toml",
        help="Path to TOML experiment config",
    )
    parser.add_argument(
        "--model-size",
        default="n",
        choices=[*MODEL_SIZES, "all"],
        help="Model size to expand via model_template (default: n)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model weights path (skips template expansion)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    print("GPU 是否可用:", torch.cuda.is_available())
    print("顯卡型號:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "無")
    print("Torch 依賴的 CUDA 版本: ", torch.version.cuda)
    print(f"Ultralytics runs project: {RUNS_PROJECT}")

    targets = resolve_model_paths(cfg, args.model_size, args.model)
    for size_label, model_path in targets:
        run_one(cfg, model_path, size_label)


if __name__ == "__main__":
    main()
