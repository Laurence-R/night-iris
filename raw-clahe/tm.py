import torch


@torch.no_grad()
def tm_reinhard(img_uint16_tensor, exposure=1.0):
    """
    使用 PyTorch CUDA 進行 16-bit 影像的 Reinhard 映射 (完美對接 GPU)
    輸入與輸出皆為 GPU Tensor
    """
    # 1. 正規化到 [0.0, 1.0] (float32)
    img_float = img_uint16_tensor.to(torch.float32) / 65535.0
    
    # 2. 使用標準 ITU-R BT.709 權重計算相對亮度 (防止色偏)
    # 假設輸入張量形狀為 (H, W, 3) 或是包含 Batch 的 (B, H, W, 3)
    r, g, b = img_float[..., 0], img_float[..., 1], img_float[..., 2]
    L_in = 0.2126 * r + 0.7152 * g + 0.0722 * b
    
    # 3. 核心 Reinhard Basic 映射
    L_scaled = L_in * exposure
    L_out = L_scaled / (1.0 + L_scaled)
    
    # 4. 計算縮放因子並套回三通道
    eps = 1e-8
    scale = L_out / (L_in + eps)
    img_out_float = img_float * scale.unsqueeze(-1)
    
    # 5. 限制範圍並轉換成 uint8 GPU Tensor
    return torch.clamp(img_out_float * 255.0, 0.0, 255.0).to(torch.uint8)


@torch.no_grad()
def tm_logarithmic(img_uint16_tensor, beta=100.0):
    """
    使用 PyTorch CUDA 進行 16-bit 影像的對數色調映射 (極速優化版)
    """

    # 1. 正規化到 [0.0, 1.0]
    img_float = img_uint16_tensor.to(torch.float32) / 65535.0
    
    # 2. 計算亮度 (同樣使用 BT.709 權重，比轉 YCrCb 快且安全)
    r, g, b = img_float[..., 0], img_float[..., 1], img_float[..., 2]
    L_in = 0.2126 * r + 0.7152 * g + 0.0722 * b
    
    # 3. GPU 對數壓縮曲線 (torch.log1p)
    beta_tensor = torch.tensor(beta, device=img_float.device, dtype=torch.float32)
    L_out = torch.log1p(beta_tensor * L_in) / torch.log1p(beta_tensor)
    L_out = torch.clamp(L_out, 0.0, 1.0)
    
    # 4. 計算縮放因子並還原三通道
    eps = 1e-8
    scale = L_out / (L_in + eps)
    img_out_float = img_float * scale.unsqueeze(-1)
    
    # 5. 限制範圍並轉換成 uint8
    return torch.clamp(img_out_float * 255.0, 0.0, 255.0).to(torch.uint8)


@torch.no_grad()
def tm_linear(img_float32_tensor, exposure=1.0):
    """
    使用 PyTorch CUDA 進行線性曝光映射
    """
    # 線性相乘直接套用在全通道上即可（不影響色相）
    img_out_float = img_float32_tensor * exposure
    
    return torch.clamp(img_out_float * 255.0, 0.0, 255.0).to(torch.uint8)


@torch.no_grad()
def tm_linear_inplace(img_float32_tensor, ldr_buffer, float_scratch, exposure=1.0):
    """
    原地線性曝光映射，輸出寫入預配置的 uint8 buffer。
    float_scratch 用於中間運算，避免每幀配置新 tensor。
    """
    torch.mul(img_float32_tensor, exposure * 255.0, out=float_scratch)
    float_scratch.clamp_(0.0, 255.0)
    ldr_buffer.copy_(float_scratch.to(torch.uint8))