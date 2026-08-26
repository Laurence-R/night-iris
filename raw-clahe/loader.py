import torch
import cv2
import numpy as np
from tm import tm_logarithmic, tm_linear

def white_balance(
    img_tensor: torch.Tensor,
    gains_buffer: torch.Tensor | None = None,
    luma_low_pct: float = 0.40,
    luma_high_pct: float = 0.95,
    gain_min: float = 0.5,
    gain_max: float = 2.0,
):
    """
    百分位遮罩 Gray-World 白平衡（原地）。

    只對 BT.709 亮度落在 [luma_low_pct, luma_high_pct] 的像素估 R/B 增益，
    避開夜間暗部綠噪與過曝高光。有效像素不足 1% 時退回整圖平均。

    輸入: (1, 3, H, W) float32 RGB，範圍 [0, 1]。
    """
    eps = 1e-5
    luma = (
        0.2126 * img_tensor[:, 0:1, :, :]
        + 0.7152 * img_tensor[:, 1:2, :, :]
        + 0.0722 * img_tensor[:, 2:3, :, :]
    )
    q = torch.quantile(
        luma.reshape(-1),
        torch.tensor([luma_low_pct, luma_high_pct], device=img_tensor.device, dtype=img_tensor.dtype),
    )
    mask = (luma >= q[0]) & (luma <= q[1])
    n_valid = mask.sum()
    n_pix = luma.numel()

    if n_valid < max(int(0.01 * n_pix), 1):
        channel_means = img_tensor.mean(dim=(2, 3), keepdim=True)
    else:
        mask_f = mask.to(img_tensor.dtype)
        denom = mask_f.sum().clamp(min=eps)
        channel_means = (img_tensor * mask_f).sum(dim=(2, 3), keepdim=True) / denom

    mean_r = torch.clamp(channel_means[:, 0:1, :, :], min=eps)
    mean_g = torch.clamp(channel_means[:, 1:2, :, :], min=eps)
    mean_b = torch.clamp(channel_means[:, 2:3, :, :], min=eps)

    gain_r = (mean_g / mean_r).clamp(gain_min, gain_max)
    gain_b = (mean_g / mean_b).clamp(gain_min, gain_max)

    if gains_buffer is None:
        gains = torch.ones_like(channel_means)
    else:
        gains = gains_buffer
        gains.fill_(1.0)
    gains[:, 0:1, :, :] = gain_r
    gains[:, 2:3, :, :] = gain_b

    img_tensor.mul_(gains)
    img_tensor.clamp_(0.0, 1.0)

def raw_to_8bit(raw_path):
    raw_img = cv2.imread(raw_path, cv2.IMREAD_UNCHANGED)
    raw_img_tensor = torch.from_numpy(raw_img).to(device=0, dtype=torch.uint16)
    raw_img_tensor = white_balance(raw_img_tensor)
    img_8bit = tm_linear(raw_img_tensor)

    return img_8bit   # return torch.uint8 tensor

def raw_to_16bit(
    raw_path,
    gpu_buffer,
    cpu_pinned_buffer: torch.Tensor | None = None,
    gains_buffer: torch.Tensor | None = None,
):
    raw_img = cv2.imread(raw_path, cv2.IMREAD_UNCHANGED)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    
    cpu_tensor = torch.from_numpy(raw_img).to(torch.float32) / 65535.0
    cpu_tensor = cpu_tensor.permute(2, 0, 1).unsqueeze(0)
    if cpu_pinned_buffer is not None:
        cpu_pinned_buffer.copy_(cpu_tensor)
        gpu_buffer.copy_(cpu_pinned_buffer, non_blocking=True)
    else:
        cpu_tensor = cpu_tensor.pin_memory()
        gpu_buffer.copy_(cpu_tensor)

    white_balance(gpu_buffer, gains_buffer)

def load_raw_batch(paths: list[str], h: int, w: int) -> np.ndarray:
    """純 CPU 批次讀取 RAW 影像，回傳 (N, H, W, 3) uint16 RGB。"""
    batch = np.empty((len(paths), h, w, 3), dtype=np.uint16)
    for i, path in enumerate(paths):
        raw_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw_img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        batch[i] = raw_img
    return batch


def batch_to_gpu(
    numpy_batch: np.ndarray,
    gpu_buffer: torch.Tensor,
    cpu_pinned_buffer: torch.Tensor,
) -> None:
    """將 (N, H, W, 3) uint16 RGB 批次轉換並一次性 H2D 至 (N, 3, H, W) float32。"""
    cpu_pinned_buffer.copy_(torch.from_numpy(numpy_batch).permute(0, 3, 1, 2))
    cpu_pinned_buffer.div_(65535.0)
    gpu_buffer.copy_(cpu_pinned_buffer, non_blocking=True)
