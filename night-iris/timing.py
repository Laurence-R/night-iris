"""Algorithm-only timers: exclude H2D/D2H. Ultralytics inference uses result.speed."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch

ALGO_STAGES: tuple[tuple[str, str], ...] = (
    ("color", "color_ms"),
    ("seg infer", "seg_infer_ms"),
    ("seg mask", "seg_mask_ms"),
    ("sem infer", "sem_infer_ms"),
    ("sem mask", "sem_mask_ms"),
    ("clahe", "clahe_ms"),
    ("fuse", "fuse_ms"),
)

INFO_STAGES: tuple[tuple[str, str], ...] = (
    ("imread", "imread_ms"),
    ("viz", "viz_ms"),
    ("seg pre", "seg_pre_ms"),
    ("seg post", "seg_post_ms"),
    ("sem pre", "sem_pre_ms"),
    ("sem post", "sem_post_ms"),
)

_STAGE_COLORS = {
    "color": "#55a868",
    "seg infer": "#4c72b0",
    "seg mask": "#64b5cd",
    "sem infer": "#8172b3",
    "sem mask": "#ccb974",
    "clahe": "#dd8452",
    "fuse": "#c44e52",
}


def yolo_speed_ms(result) -> tuple[float, float, float]:
    """Return (preprocess, inference, postprocess) ms. Only inference counts as algo."""
    speed = getattr(result, "speed", None) or {}
    return (
        float(speed.get("preprocess", 0.0) or 0.0),
        float(speed.get("inference", 0.0) or 0.0),
        float(speed.get("postprocess", 0.0) or 0.0),
    )


def cpu_ms(fn, *args, **kwargs):
    t0 = perf_counter()
    out = fn(*args, **kwargs)
    return out, (perf_counter() - t0) * 1000.0


def cuda_kernel_ms(device: torch.device, fn, *args, **kwargs):
    """Time a GPU kernel after inputs are already on device; no copy in fn."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return cpu_ms(fn, *args, **kwargs)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    return out, float(start.elapsed_time(end))


@dataclass
class StageTiming:
    image: str
    warmup: bool
    color_ms: float
    seg_infer_ms: float
    seg_mask_ms: float
    sem_infer_ms: float
    sem_mask_ms: float
    clahe_ms: float
    fuse_ms: float
    imread_ms: float
    viz_ms: float
    seg_pre_ms: float
    seg_post_ms: float
    sem_pre_ms: float
    sem_post_ms: float

    @property
    def algo_ms(self) -> float:
        return (
            self.color_ms
            + self.seg_infer_ms
            + self.seg_mask_ms
            + self.sem_infer_ms
            + self.sem_mask_ms
            + self.clahe_ms
            + self.fuse_ms
        )


def _values(rows: list[StageTiming], key: str) -> np.ndarray:
    if key == "algo_ms":
        return np.array([r.algo_ms for r in rows], dtype=np.float64)
    return np.array([getattr(r, key) for r in rows], dtype=np.float64)


def _ms1(x: float) -> float:
    return round(float(x), 1)


def summarize(rows: list[StageTiming], keys: tuple[str, ...], title: str | None = None) -> None:
    measured = [r for r in rows if not r.warmup]
    if not measured:
        print(">> No non-warmup timings (raise n or lower timing.warmup)")
        return
    label = title or "Algorithm latency (exclude H2D/D2H, imread, viz)"
    print(f">> {label}  n={len(measured)}")
    for key in keys:
        vals = _values(measured, key)
        print(
            f"   {key:16s}  mean={_ms1(vals.mean()):5.1f}  "
            f"p50={_ms1(np.percentile(vals, 50)):5.1f}  "
            f"p99={_ms1(np.percentile(vals, 99)):5.1f} ms"
        )


_PER_IMAGE_MAX = 12
_SLOWEST_K = 8


def _short_name(name: str, limit: int = 28) -> str:
    if len(name) <= limit:
        return name
    return name[: limit - 1] + "…"


def write_timings_charts(path: Path, rows: list[StageTiming]) -> Path:
    """Fixed-size summary PNG. Per-image stacked bars only when n is small."""
    measured = [r for r in rows if not r.warmup]
    if not measured:
        raise SystemExit("No non-warmup timings to plot (raise n or lower timing.warmup)")

    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Arial", "Helvetica"]
    plt.rcParams["axes.unicode_minus"] = False

    n = len(measured)
    algo_keys = ["algo_ms"] + [key for _, key in ALGO_STAGES]
    algo_labels = ["algo"] + [name for name, _ in ALGO_STAGES]
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 13.5), dpi=140)

    ax = axes[0]
    if n <= _PER_IMAGE_MAX:
        names = [_short_name(r.image, 18) for r in measured]
        bottom = np.zeros(n, dtype=np.float64)
        for label, key in ALGO_STAGES:
            vals = np.array([_ms1(v) for v in _values(measured, key)])
            ax.bar(
                names,
                vals,
                bottom=bottom,
                label=label,
                color=_STAGE_COLORS[label],
                edgecolor="black",
                linewidth=0.4,
                width=0.62,
            )
            bottom += vals
        for i, total in enumerate(bottom):
            ax.text(i, total + 0.4, f"{_ms1(total):.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"Per-image algorithm latency (n={n}, warmup excluded)")
        ax.set_ylabel("Latency (ms)")
        ax.set_ylim(0, max(float(bottom.max()) * 1.18, 1.0))
        ax.legend(loc="upper right", ncol=4, fontsize=8, frameon=False)
        ax.tick_params(axis="x", labelrotation=25)
    else:
        ys = _values(measured, "algo_ms")
        xs = np.arange(1, n + 1)
        ax.scatter(xs, ys, s=12, c="#4c72b0", alpha=0.7, linewidths=0)
        p50_y = float(np.percentile(ys, 50))
        p99_y = float(np.percentile(ys, 99))
        ax.axhline(p50_y, color="#4c72b0", linestyle="--", linewidth=1.0, label="P50")
        ax.axhline(p99_y, color="#dd8452", linestyle="--", linewidth=1.0, label="P99")
        cutoff = float(np.percentile(ys, 95))
        slow_idx = [int(i) for i in np.argsort(ys)[::-1] if ys[int(i)] >= cutoff][:5]
        for rank, i in enumerate(slow_idx):
            ax.annotate(
                _short_name(measured[i].image, 22),
                (xs[i], ys[i]),
                textcoords="offset points",
                xytext=(6, 8 + rank * 9),
                fontsize=7,
                arrowprops={"arrowstyle": "-", "color": "#444444", "lw": 0.5},
            )
        ax.set_title(f"algo_ms by image index (n={n}, warmup excluded)")
        ax.set_xlabel("Image index")
        ax.set_ylabel("Latency (ms)")
        ax.set_ylim(0, max(float(ys.max()) * 1.18, 1.0))
        ax.legend(loc="upper right", frameon=False)

    ax = axes[1]
    x = np.arange(len(algo_keys))
    p50 = np.array([_ms1(np.percentile(_values(measured, k), 50)) for k in algo_keys])
    p99 = np.array([_ms1(np.percentile(_values(measured, k), 99)) for k in algo_keys])
    w = 0.36
    bars_p50 = ax.bar(x - w / 2, p50, w, label="P50", color="#4c72b0", edgecolor="black", linewidth=0.4)
    bars_p99 = ax.bar(x + w / 2, p99, w, label="P99", color="#dd8452", edgecolor="black", linewidth=0.4)
    ax.bar_label(bars_p50, fmt="%.1f", fontsize=8, padding=2)
    ax.bar_label(bars_p99, fmt="%.1f", fontsize=8, padding=2)
    ax.set_xticks(x, algo_labels)
    ax.set_title("Algorithm stages  P50 / P99")
    ax.set_ylabel("Latency (ms)")
    ax.set_ylim(0, max(float(p99.max()) * 1.22, 1.0))
    ax.legend(loc="upper right", frameon=False)

    ax = axes[2]
    box_data = [_values(measured, k) for k in algo_keys]
    bp = ax.boxplot(box_data, tick_labels=algo_labels, showfliers=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#d9e3f0")
        patch.set_edgecolor("black")
        patch.set_linewidth(0.6)
    ax.set_title("Algorithm stage distribution")
    ax.set_ylabel("Latency (ms)")

    ax = axes[3]
    info_labels = [name for name, _ in INFO_STAGES]
    info_keys = [key for _, key in INFO_STAGES]
    x = np.arange(len(info_keys))
    info_p50 = np.array([_ms1(np.percentile(_values(measured, k), 50)) for k in info_keys])
    bars = ax.bar(x, info_p50, 0.55, color="#8c8c8c", edgecolor="black", linewidth=0.4)
    ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set_xticks(x, info_labels)
    ax.set_title("Not in algo_ms  (I/O, viz, YOLO pre/post)  P50")
    ax.set_ylabel("Latency (ms)")
    ax.set_ylim(0, max(float(info_p50.max()) * 1.22, 1.0))
    slow = sorted(measured, key=lambda r: r.algo_ms, reverse=True)[: min(_SLOWEST_K, n)]
    slow_lines = "\n".join(f"{r.algo_ms:6.1f} ms  {_short_name(r.image, 48)}" for r in slow)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    fig.text(
        0.02,
        0.01,
        f"Slowest algo_ms (up to {_SLOWEST_K}):\n{slow_lines}",
        fontsize=8,
        family="monospace",
        va="bottom",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
