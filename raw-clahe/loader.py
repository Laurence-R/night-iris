import torch
import cv2
import numpy as np
from tm import tm_logarithmic, tm_linear

def white_balance(img_tensor: torch.Tensor, gains_buffer: torch.Tensor | None = None):
    """
    在 GPU 上對 (1, 3, H, W) 的浮點數張量進行『原地（In-place）灰色世界白平衡』。
    此函式直接修改輸入的張量（帶底線算子），達到零記憶體碎片與極限加速。
    
    參數:
        img_tensor (torch.Tensor): 位於 GPU 上的張量，形狀為 (1, 3, H, W)，數值範圍為 [0.0, 1.0]
        
    通道順序說明 (以 RAW 轉出來的 RGB 為例):
        通道 0 = R, 通道 1 = G, 通道 2 = B (若為 BGR 則對調即可，原理相同)
    """
    # 1. 計算每個通道的平均值 (對 H 維度 dim=2 與 W 維度 dim=3 取平均)
    # keepdim=True 會保持形狀為 (1, 3, 1, 1)，完美對齊廣播機制 (Broadcasting)
    channel_means = img_tensor.mean(dim=(2, 3), keepdim=True) # 形狀: (1, 3, 1, 1)
    
    mean_r = channel_means[:, 0:1, :, :]
    mean_g = channel_means[:, 1:2, :, :]
    mean_b = channel_means[:, 2:3, :, :]
    
    # 2. 防止分母為 0 (避免全黑圖片除以零錯誤)
    eps = 1e-5
    mean_r = torch.clamp(mean_r, min=eps)
    mean_b = torch.clamp(mean_b, min=eps)
    
    # 3. 以 G 通道為基準計算 R 與 B 的補償增益
    gain_r = mean_g / mean_r
    gain_b = mean_g / mean_b
    
    # 4. 重用預配置增益 buffer，避免每幀配置新 tensor
    if gains_buffer is None:
        gains = torch.ones_like(channel_means)
    else:
        gains = gains_buffer
        gains.fill_(1.0)
    gains[:, 0:1, :, :] = gain_r
    gains[:, 2:3, :, :] = gain_b
    
    # 5. 原地乘法與截斷 (利用 .mul_() 與 .clamp_()，零顯存分配)
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
