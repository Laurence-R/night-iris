import argparse
import os
import sys
import tomllib
from pathlib import Path

if sys.platform == "win32":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
else:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,expandable_segments:True"

import cv2
import numpy as np
import torch
import pandas as pd
import gc
import random as rd

from loader import load_raw_batch, batch_to_gpu, range_spec
from clahe import apply_clahe_korina, ClaheWorkspace, _ColorKernels
from tm import tm_linear_inplace
from visualize import plot_scatter, plot_latency_breakdown
import kornia.filters as KF


def _event():
    return torch.cuda.Event(enable_timing=True)


def load_config(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def write_run_config(dest_path: str, cfg: dict) -> None:
    paths = cfg["paths"]
    image = cfg["image"]
    clahe = cfg["clahe"]
    bilateral = cfg["bilateral"]
    grid = clahe["grid"]
    kernel = bilateral["kernel"]
    sigma_space = bilateral["sigma_space"]

    text = (
        f'[paths]\n'
        f'input_dir = "{paths["input_dir"]}"\n'
        f'output_dir = "{paths["output_dir"]}"\n'
        f'\n'
        f'[image]\n'
        f'height = {image["height"]}\n'
        f'width = {image["width"]}\n'
        f'count = {image["count"]}\n'
        f'batch_size = {image["batch_size"]}\n'
        f'warmup = {image["warmup"]}\n'
        f'dynamic_range = "{image["dynamic_range"]}"\n'
        f'\n'
        f'[clahe]\n'
        f'grid = [{grid[0]}, {grid[1]}]\n'
        f'clip_limit = {clahe["clip_limit"]}\n'
        f'\n'
        f'[bilateral]\n'
        f'enabled = {"true" if bilateral["enabled"] else "false"}\n'
        f'kernel = [{kernel[0]}, {kernel[1]}]\n'
        f'sigma_color = {bilateral["sigma_color"]}\n'
        f'sigma_space = [{sigma_space[0]}, {sigma_space[1]}]\n'
    )
    out = Path(dest_path) / "run_config.toml"
    out.write_text(text, encoding="utf-8")
    print(f">> Run config saved to: {out}")


def enhance_processing(items, dest_path, cfg: dict):
    image = cfg["image"]
    clahe_cfg = cfg["clahe"]
    bilateral = cfg["bilateral"]

    h = int(image["height"])
    w = int(image["width"])
    batch_size = int(image["batch_size"])
    warmup_count = min(int(image["warmup"]), len(items))
    use_bilateral = bool(bilateral["enabled"])
    bf_kernel = tuple(int(x) for x in bilateral["kernel"])
    bf_sigma_c = float(bilateral["sigma_color"])
    bf_sigma_s = tuple(float(x) for x in bilateral["sigma_space"])
    grid_size = tuple(int(x) for x in clahe_cfg["grid"])
    clip_limit = float(clahe_cfg["clip_limit"])
    dynamic_range = cfg["image"]["dynamic_range"]
    _, scale = range_spec(dynamic_range)

    print(f"=== Loading all images into memory ({dynamic_range}, scale {scale:g})... ===")
    all_numpy = load_raw_batch(items, h, w, dynamic_range)
    print(f"Loaded {len(items)} images.\n")

    yuv_buffer = torch.zeros((1, 3, h, w), dtype=torch.float32, device="cuda")
    y_enhanced_buffer = torch.zeros((1, 1, h, w), dtype=torch.float32, device="cuda")
    rgb_out = torch.zeros((h, w, 3), dtype=torch.float32, device="cuda")
    tm_float_scratch = torch.empty((h, w, 3), dtype=torch.float32, device="cuda")
    clahe_workspace = ClaheWorkspace(h, w, grid_size=grid_size, device="cuda")
    color_kernels = _ColorKernels(device="cuda", dtype=torch.float32)

    bf_buf = torch.empty((1, 3, h, w), dtype=torch.float32, device="cuda") if use_bilateral else None

    gpu_batch = torch.zeros((batch_size, 3, h, w), dtype=torch.float32, device="cuda")
    cpu_pinned = torch.empty((batch_size, 3, h, w), dtype=torch.float32).pin_memory()
    out_gpu = torch.empty((batch_size, h, w, 3), dtype=torch.uint8, device="cuda")
    out_pinned = torch.empty((batch_size, h, w, 3), dtype=torch.uint8).pin_memory()

    # 全部結果先留在一般（非 pinned）記憶體，跑完所有 chunk 才統一落盤，
    # 讓 chunk 與 chunk 之間 GPU 不會因等待磁碟寫入而閒置降頻。
    all_results = np.empty((len(items), h, w, 3), dtype=np.uint8)

    def process_slot(i):
        apply_clahe_korina(
            gpu_batch[i : i + 1],
            yuv_buffer,
            y_enhanced_buffer,
            rgb_out,
            clahe_workspace,
            color_kernels,
            clip_limit=clip_limit,
        )
        if use_bilateral:
            bf_buf.copy_(rgb_out.permute(2, 0, 1).unsqueeze(0))
            rgb_out.copy_(
                KF.bilateral_blur(bf_buf, bf_kernel, bf_sigma_c, bf_sigma_s)
                .squeeze(0)
                .permute(1, 2, 0)
            )
        tm_linear_inplace(rgb_out, out_gpu[i], tm_float_scratch)

    print(f"=== Warmup ({warmup_count} images)... ===")
    batch_to_gpu(
        all_numpy[:warmup_count], gpu_batch[:warmup_count], cpu_pinned[:warmup_count], scale
    )
    for i in range(warmup_count):
        process_slot(i)
    torch.cuda.synchronize()
    print("Warmup done.\n")

    # 每個 chunk 內的所有 GPU 工作（H2D、CLAHE、TM、D2H）連續排入同一個 stream，
    # 只在 chunk 結尾做一次 synchronize，避免逐張同步造成 GPU 頻繁閒置、降頻。
    records = []
    count = 0
    for chunk_start in range(0, len(items), batch_size):
        chunk_items = items[chunk_start : chunk_start + batch_size]
        n = len(chunk_items)

        h2d_start, h2d_end = _event(), _event()
        h2d_start.record()
        batch_to_gpu(
            all_numpy[chunk_start : chunk_start + n], gpu_batch[:n], cpu_pinned[:n], scale
        )
        h2d_end.record()

        clahe_events = [_event() for _ in range(n)]
        bf_events = [_event() for _ in range(n)] if use_bilateral else None
        tm_events = [_event() for _ in range(n)]
        for i in range(n):
            apply_clahe_korina(
                gpu_batch[i : i + 1],
                yuv_buffer,
                y_enhanced_buffer,
                rgb_out,
                clahe_workspace,
                color_kernels,
                clip_limit=clip_limit,
            )
            clahe_events[i].record()
            if use_bilateral:
                bf_buf.copy_(rgb_out.permute(2, 0, 1).unsqueeze(0))
                rgb_out.copy_(
                    KF.bilateral_blur(bf_buf, bf_kernel, bf_sigma_c, bf_sigma_s)
                    .squeeze(0)
                    .permute(1, 2, 0)
                )
                bf_events[i].record()
            tm_linear_inplace(rgb_out, out_gpu[i], tm_float_scratch)
            tm_events[i].record()

        d2h_start, d2h_end = _event(), _event()
        d2h_start.record()
        out_pinned[:n].copy_(out_gpu[:n], non_blocking=True)
        d2h_end.record()

        torch.cuda.synchronize()

        all_results[chunk_start : chunk_start + n] = out_pinned[:n].numpy()

        h2d_per = h2d_start.elapsed_time(h2d_end) / n
        d2h_per = d2h_start.elapsed_time(d2h_end) / n
        print(
            f"=== Chunk [{chunk_start}:{chunk_start + n}] "
            f"H2D(total)={h2d_per * n:.2f}ms D2H(total)={d2h_per * n:.2f}ms ==="
        )

        prev_event = h2d_end
        for i, item in enumerate(chunk_items):
            clahe_time = prev_event.elapsed_time(clahe_events[i])
            if use_bilateral:
                bf_time = clahe_events[i].elapsed_time(bf_events[i])
                tm_time = bf_events[i].elapsed_time(tm_events[i])
            else:
                bf_time = 0.0
                tm_time = clahe_events[i].elapsed_time(tm_events[i])
            prev_event = tm_events[i]

            bf_str = f"  BF={bf_time:.3f}ms" if use_bilateral else ""
            print(
                f"{os.path.basename(item)}  "
                f"H2D={h2d_per:.3f}ms  CLAHE={clahe_time:.3f}ms{bf_str}  "
                f"TM={tm_time:.3f}ms  D2H={d2h_per:.3f}ms"
            )

            records.append(
                {
                    "Number": count,
                    "Image_ID": item,
                    "Loading_Time": h2d_per,
                    "CLAHE_Time": clahe_time,
                    "BF_Time": bf_time,
                    "ToneMapping_Time": tm_time,
                    "D2H_Time": d2h_per,
                    "Total_Preprocessing": clahe_time + bf_time + tm_time,
                    "Total_Pipeline": h2d_per + clahe_time + bf_time + tm_time + d2h_per,
                }
            )
            count += 1

    del all_numpy, gpu_batch, cpu_pinned, out_gpu, out_pinned

    processed_dir = os.path.join(dest_path, "processed-img")
    latency_dir = os.path.join(dest_path, "latency")
    print("=== GPU processing done. Writing PNGs to disk... ===")
    for item, result in zip(items, all_results):
        bgr_ldr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(processed_dir, os.path.basename(item)), bgr_ldr)
    del all_results

    print(f"\nDone. Results saved to: {processed_dir}")
    write_run_config(dest_path, cfg)

    df = pd.DataFrame(records)
    plot_scatter(df, latency_dir)
    plot_latency_breakdown(df, latency_dir)

    p99 = df["Total_Preprocessing"].quantile(0.99)
    spikes = (df["Total_Preprocessing"] > 30).sum()
    print(f"\n>> P99 CLAHE 管線延遲: {p99:.2f} ms")
    print(f">> 超過 30 ms 的尖峰次數: {spikes} / {len(df)}")


def parse_args():
    parser = argparse.ArgumentParser(description="GPU CLAHE kernel test and latency measurement")
    parser.add_argument(
        "--config",
        default="configs/default.toml",
        help="Path to TOML config (default: configs/default.toml)",
    )
    parser.add_argument("--input", default=None, help="Override input directory")
    parser.add_argument("--output", default=None, help="Override output directory")
    parser.add_argument("--count", type=int, default=None, help="Override image count")
    parser.add_argument(
        "--no-bilateral",
        action="store_true",
        help="Disable bilateral filter",
    )
    parser.add_argument(
        "--dynamic-range",
        choices=["ldr", "hdr"],
        default=None,
        help="Input dynamic range: ldr (8-bit, /255) or hdr (16-bit, /65535). Overrides config.",
    )
    return parser.parse_args()


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.input is not None:
        cfg["paths"]["input_dir"] = args.input
    if args.output is not None:
        cfg["paths"]["output_dir"] = args.output
    if args.count is not None:
        cfg["image"]["count"] = args.count
    if args.no_bilateral:
        cfg["bilateral"]["enabled"] = False
    if args.dynamic_range is not None:
        cfg["image"]["dynamic_range"] = args.dynamic_range
    dynamic_range = cfg["image"].get("dynamic_range")
    if dynamic_range not in ("ldr", "hdr"):
        raise SystemExit("image.dynamic_range must be 'ldr' or 'hdr' (or pass --dynamic-range)")
    return cfg


def main():
    args = parse_args()
    print(f"CUDA available: {torch.cuda.is_available()}")
    cfg = apply_cli_overrides(load_config(args.config), args)

    raw_path = cfg["paths"]["input_dir"]
    result_path = cfg["paths"]["output_dir"]
    img_count = int(cfg["image"]["count"])

    os.makedirs(os.path.join(result_path, "processed-img"), exist_ok=True)
    os.makedirs(os.path.join(result_path, "latency"), exist_ok=True)

    gc.disable()
    torch.backends.cudnn.benchmark = True

    items = [os.path.join(raw_path, f) for f in os.listdir(raw_path)][:img_count]
    rd.shuffle(items)

    enhance_processing(items=items, dest_path=result_path, cfg=cfg)


if __name__ == "__main__":
    main()
