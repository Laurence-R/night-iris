import argparse
import json
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import torch
from ultralytics import YOLO

MODEL_SIZES = ("n", "s", "m", "l", "x")


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


def experiment_dir(name: str, size_label: str) -> Path:
    path = Path("experiments") / f"{name}_{size_label}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_snapshot(exp_dir: Path, cfg: dict, model_path: str, size_label: str) -> None:
    train = cfg.get("train", {})
    val = cfg.get("val", {})
    predict = cfg.get("predict", {})
    export = cfg.get("export", {})
    quantize = export.get("quantize", 16)

    text = (
        f'name = "{cfg["name"]}_{size_label}"\n'
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
        f'source = "{predict.get("source", "dark_rgb")}"\n'
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
    t = cfg.get("train", {})
    model.train(
        data=cfg["data"],
        epochs=t.get("epochs", 50),
        patience=t.get("patience", 15),
        batch=t.get("batch", 8),
        imgsz=t.get("imgsz", 640),
        device=cfg.get("device", 0),
    )


def predict_model(model: YOLO, cfg: dict) -> None:
    p = cfg.get("predict", {})
    model.predict(
        source=p.get("source", "dark_rgb"),
        conf=p.get("conf", 0.25),
        batch=p.get("batch", 1),
        save=True,
        device=cfg.get("device", 0),
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


def validate_model(model: YOLO, cfg: dict, model_path: str, size_label: str) -> dict:
    v = cfg.get("val", {})
    batch = int(v.get("batch", 1))
    imgsz = int(v.get("imgsz", 640))
    print(f"Validating with batch={batch} imgsz={imgsz}")
    results = model.val(
        data=cfg["data"],
        device=cfg.get("device", 0),
        batch=batch,
        imgsz=imgsz,
    )

    preprocess_time = results.speed["preprocess"]
    inference_time = results.speed["inference"]
    postprocess_time = results.speed["postprocess"]
    total_time_per_image = preprocess_time + inference_time + postprocess_time
    fps = 1000 / total_time_per_image if total_time_per_image > 0 else 0.0

    map50 = float(results.box.map50) if hasattr(results.box, "map50") else None
    map50_95 = float(results.box.map)

    print("--- 驗證結果與平均單張耗時 ---")
    if map50 is not None:
        print(f"精確度 mAP50: {map50:.4f}")
    print(f"精確度 mAP50-95: {map50_95:.4f}")
    print(f"batch: {batch}")
    print(f"平均前處理: {preprocess_time:.2f} ms / 張")
    print(f"平均模型推論: {inference_time:.2f} ms / 張")
    print(f"平均後處理: {postprocess_time:.2f} ms / 張")
    print(f"單張總耗時: {total_time_per_image:.2f} ms")
    print(f"等效吞吐量 (FPS): {fps:.2f}")

    metrics = {
        "name": f"{cfg['name']}_{size_label}",
        "mode": "val",
        "model": model_path,
        "data": cfg["data"],
        "device": cfg.get("device", 0),
        "batch": batch,
        "imgsz": imgsz,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "map50": map50,
        "map50_95": map50_95,
        "preprocess_ms": preprocess_time,
        "inference_ms": inference_time,
        "postprocess_ms": postprocess_time,
        "total_ms": total_time_per_image,
        "fps": fps,
    }

    exp_dir = experiment_dir(cfg["name"], size_label)
    write_snapshot(exp_dir, cfg, model_path, size_label)
    metrics_path = exp_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f">> Metrics saved to: {metrics_path}")
    return metrics


def run_one(cfg: dict, model_path: str, size_label: str) -> None:
    if not Path(model_path).exists():
        raise SystemExit(f"Model not found: {model_path}")

    print(f"\n=== Experiment {cfg['name']}_{size_label} | mode={cfg['mode']} | model={model_path} ===")
    model = YOLO(model_path)
    mode = cfg["mode"]

    if mode == "train":
        exp_dir = experiment_dir(cfg["name"], size_label)
        write_snapshot(exp_dir, cfg, model_path, size_label)
        train_model(model, cfg)
    elif mode == "val":
        validate_model(model, cfg, model_path, size_label)
    elif mode == "predict":
        exp_dir = experiment_dir(cfg["name"], size_label)
        write_snapshot(exp_dir, cfg, model_path, size_label)
        predict_model(model, cfg)
    elif mode == "export":
        exp_dir = experiment_dir(cfg["name"], size_label)
        write_snapshot(exp_dir, cfg, model_path, size_label)
        export_model(model, cfg)
    else:
        raise SystemExit(f"Unknown mode: {mode!r} (expected train|val|predict|export)")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO26 ablation experiment runner")
    parser.add_argument(
        "--config",
        default="configs/baseline_val.toml",
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

    targets = resolve_model_paths(cfg, args.model_size, args.model)
    for size_label, model_path in targets:
        run_one(cfg, model_path, size_label)


if __name__ == "__main__":
    main()
