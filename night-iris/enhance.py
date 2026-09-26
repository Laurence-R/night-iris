"""Full-frame rectangular CLAHE (gpu-clahe kernel) plus dark-mask write-back."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from timing import cuda_kernel_ms


def _load_gpu_clahe():
    path = Path(__file__).resolve().parent.parent / "gpu-clahe" / "clahe.py"
    spec = importlib.util.spec_from_file_location("gpu_clahe_kernel", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load CLAHE kernel from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_clahe = _load_gpu_clahe()
ClaheWorkspace = _clahe.ClaheWorkspace
_ColorKernels = _clahe._ColorKernels
apply_clahe_korina = _clahe.apply_clahe_korina


class EnhanceEngine:
    def __init__(self, grid: tuple[int, int], clip_limit: float, device: str):
        self.grid = grid
        self.clip_limit = clip_limit
        self.device = torch.device(device if device != "cpu" else "cpu")
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        self.kernels = _ColorKernels(device=self.device, dtype=torch.float32)
        self._cache: dict[tuple, tuple] = {}

    def _buffers(self, h: int, w: int):
        key = (h, w, self.grid, str(self.device))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        ws = ClaheWorkspace(h, w, grid_size=self.grid, device=self.device, dtype=torch.float32)
        yuv = torch.zeros((1, 3, h, w), dtype=torch.float32, device=self.device)
        y_enh = torch.zeros((1, 1, h, w), dtype=torch.float32, device=self.device)
        rgb_out = torch.zeros((h, w, 3), dtype=torch.float32, device=self.device)
        work = torch.zeros((1, 3, h, w), dtype=torch.float32, device=self.device)
        pack = (ws, yuv, y_enh, rgb_out, work)
        self._cache[key] = pack
        return pack

    @torch.no_grad()
    def clahe_rgb(self, rgb: np.ndarray) -> tuple[np.ndarray, float]:
        """rgb uint8 (H,W,3) → (CLAHE uint8, kernel_ms). kernel_ms excludes H2D/D2H."""
        h, w = rgb.shape[:2]
        ws, yuv, y_enh, rgb_out, work = self._buffers(h, w)
        cpu = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        work.copy_(cpu.to(self.device, non_blocking=True))
        if work.is_cuda:
            torch.cuda.synchronize()

        def _kernel():
            apply_clahe_korina(work, yuv, y_enh, rgb_out, ws, self.kernels, clip_limit=self.clip_limit)

        _, clahe_ms = cuda_kernel_ms(self.device, _kernel)
        out = rgb_out.clamp(0.0, 1.0).mul(255.0).to(torch.uint8).cpu().numpy()
        return out, clahe_ms
