import torch
import cv2
import numpy as np

# CLAHE 的輸入是 [0, 1] float32。LDR 與 HDR 只差整數深度和正規化分母。
_DYNAMIC_RANGE = {
    "ldr": (np.uint8, 255.0),
    "hdr": (np.uint16, 65535.0),
}


def range_spec(dynamic_range: str) -> tuple[np.dtype, float]:
    try:
        return _DYNAMIC_RANGE[dynamic_range]
    except KeyError:
        raise ValueError(
            f"dynamic_range must be 'ldr' or 'hdr', got {dynamic_range!r}"
        ) from None


def load_raw_batch(paths: list[str], h: int, w: int, dynamic_range: str) -> np.ndarray:
    """Read images with OpenCV IMREAD_UNCHANGED. Returns (N, H, W, 3) integer RGB."""
    dtype, _ = range_spec(dynamic_range)
    batch = np.empty((len(paths), h, w, 3), dtype=dtype)
    for i, path in enumerate(paths):
        raw_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw_img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        if raw_img.dtype != dtype:
            raise ValueError(
                f"{path} is {raw_img.dtype}, expected {dtype.__name__} for dynamic_range={dynamic_range}"
            )
        if raw_img.ndim != 3 or raw_img.shape[2] != 3:
            raise ValueError(f"{path} must be 3-channel, got shape {raw_img.shape}")
        if raw_img.shape[0] != h or raw_img.shape[1] != w:
            raise ValueError(
                f"{path} is {raw_img.shape[1]}x{raw_img.shape[0]}, expected {w}x{h}"
            )
        batch[i] = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    return batch


def batch_to_gpu(
    numpy_batch: np.ndarray,
    gpu_buffer: torch.Tensor,
    cpu_pinned_buffer: torch.Tensor,
    scale: float,
) -> None:
    """將 (N, H, W, 3) 整數 RGB 正規化到 [0, 1]，一次 H2D 到 (N, 3, H, W) float32。"""
    cpu_pinned_buffer.copy_(torch.from_numpy(numpy_batch).permute(0, 3, 1, 2))
    cpu_pinned_buffer.div_(scale)
    gpu_buffer.copy_(cpu_pinned_buffer, non_blocking=True)
