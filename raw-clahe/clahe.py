"""
CLAHE for 16-bit RGB images using PyTorch on GPU.
Uses pre-allocated workspace buffers to avoid per-frame CUDA allocations.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


class _ColorKernels:
    """Cached conv2d weights for RGB <-> YUV (BT.470-5)."""

    def __init__(self, device: torch.device, dtype: torch.dtype):
        rgb2yuv = torch.tensor(
            [
                [0.299, 0.587, 0.114],
                [-0.147, -0.289, 0.436],
                [0.615, -0.515, -0.100],
            ],
            device=device,
            dtype=dtype,
        )
        yuv2rgb = torch.tensor(
            [
                [1.0, 0.0, 1.14],
                [1.0, -0.396, -0.581],
                [1.0, 2.029, 0.0],
            ],
            device=device,
            dtype=dtype,
        )
        self.rgb2yuv_weight = rgb2yuv.view(3, 3, 1, 1)
        self.yuv2rgb_weight = yuv2rgb.view(3, 3, 1, 1)


class ClaheWorkspace:
    """Pre-allocated buffers for fixed-size single-channel CLAHE."""

    def __init__(
        self,
        height: int,
        width: int,
        grid_size: Tuple[int, int] = (8, 8),
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.height = height
        self.width = width
        self.grid_size = grid_size
        self.device = torch.device(device)
        self.dtype = dtype

        kernel_vert = math.ceil(height / grid_size[0])
        kernel_horz = math.ceil(width / grid_size[1])
        if kernel_vert % 2:
            kernel_vert += 1
        if kernel_horz % 2:
            kernel_horz += 1

        self.kernel_vert = kernel_vert
        self.kernel_horz = kernel_horz
        self.gh, self.gw = grid_size
        self.gh2, self.gw2 = grid_size[0] * 2, grid_size[1] * 2
        self.th2, self.tw2 = kernel_vert // 2, kernel_horz // 2
        self.num_tiles = grid_size[0] * grid_size[1]

        pad_vert = kernel_vert * grid_size[0] - height
        pad_horz = kernel_horz * grid_size[1] - width
        self.pad_vert = pad_vert
        self.pad_horz = pad_horz
        self.padded_h = height + pad_vert
        self.padded_w = width + pad_horz

        self.hist_tiles = torch.empty(
            (1, self.gh, self.gw, 1, kernel_vert, kernel_horz), device=device, dtype=dtype
        )
        self.img_padded = torch.empty((1, 1, self.padded_h, self.padded_w), device=device, dtype=dtype)
        self.interp_tiles = torch.empty(
            (1, self.gh2, self.gw2, 1, self.th2, self.tw2), device=device, dtype=dtype
        )
        self.luts = torch.empty((1, self.gh, self.gw, 1, 256), device=device, dtype=dtype)
        self.mapped_luts = torch.empty(
            (1, self.gh2, self.gw2, 4, 1, 256), device=device, dtype=dtype
        )
        self.tiles_equalized = torch.empty(
            (1, self.gh2, self.gw2, 1, self.th2, self.tw2), device=device, dtype=dtype
        )
        self.preinterp_tiles = torch.empty(
            (1, self.gh2, self.gw2, 4, 1, self.th2, self.tw2), device=device, dtype=dtype
        )
        self.flatten_indices = torch.empty(
            (1, self.gh2, self.gw2, 4, 1, self.th2 * self.tw2), device=device, dtype=torch.long
        )
        self.eq_output = torch.empty((1, 1, self.padded_h, self.padded_w), device=device, dtype=dtype)
        self.histos = torch.empty((self.num_tiles, 256), device=device, dtype=dtype)
        self.tiles_flat = torch.empty((self.num_tiles, kernel_vert * kernel_horz), device=device, dtype=dtype)

        _pixels = kernel_vert * kernel_horz
        _thw    = self.th2 * self.tw2
        self.hist_indices_f = torch.empty((self.num_tiles, _pixels), device=device, dtype=dtype)
        self.hist_indices   = torch.empty((self.num_tiles, _pixels), device=device, dtype=torch.long)
        self.ones_flat      = torch.ones( (self.num_tiles, _pixels), device=device, dtype=dtype)
        self.v_range        = torch.arange(256, device=device, dtype=dtype)
        self.luts_flat_buf  = torch.empty((self.num_tiles, 256), device=device, dtype=dtype)
        self._flatten_f     = torch.empty((1, self.gh2, self.gw2, 1, self.th2, self.tw2), device=device, dtype=dtype)
        self._flatten_i     = torch.empty((1, self.gh2, self.gw2, 1, _thw), device=device, dtype=torch.long)
        self._t_buf         = torch.empty((1, self.gh2 - 2, self.gw2 - 2, 1, self.th2, self.tw2), device=device, dtype=dtype)
        self._b_buf         = torch.empty_like(self._t_buf)
        self._sub_buf       = torch.empty_like(self._t_buf)
        self._edge_buf      = torch.empty((1, max(self.gh2 - 2, self.gw2 - 2), 1, self.th2, self.tw2), device=device, dtype=dtype)
        self._perm_buf      = torch.empty((1, 1, self.gh2, self.th2, self.gw2, self.tw2), device=device, dtype=dtype)

        self._init_interp_weights()
        self._init_lut_indices()

    def _init_interp_weights(self) -> None:
        th, tw = self.th2, self.tw2
        gh2, gw2 = self.gh2, self.gw2

        ih = (
            torch.arange(2 * th - 1, -1, -1, dtype=self.dtype, device=self.device)
            .div(2.0 * th - 1)[None]
            .transpose(-2, -1)
            .expand(2 * th, tw)
        )
        ih = ih.unfold(0, th, th).unfold(1, tw, tw)

        iw = (
            torch.arange(2 * tw - 1, -1, -1, dtype=self.dtype, device=self.device)
            .div(2.0 * tw - 1)
            .expand(th, 2 * tw)
        )
        iw = iw.unfold(0, th, th).unfold(1, tw, tw)

        self.tiw = iw.expand((gw2 - 2) // 2, 2, th, tw).reshape(gw2 - 2, 1, th, tw).unsqueeze(0)
        self.tih = ih.repeat((gh2 - 2) // 2, 1, 1, 1).unsqueeze(1)

    def _init_lut_indices(self) -> None:
        gh2, gw2 = self.gh2, self.gw2

        if gh2 > 2:
            j_floor = torch.arange(1, gh2 - 1, device=self.device).view(gh2 - 2, 1).div(2, rounding_mode="trunc")
            j_idxs = torch.tensor([[0, 0, 1, 1], [-1, -1, 0, 0]] * ((gh2 - 2) // 2), device=self.device)
            self.j_idxs = j_idxs + j_floor
        else:
            self.j_idxs = torch.empty(0, 4, dtype=torch.long, device=self.device)

        if gw2 > 2:
            i_floor = torch.arange(1, gw2 - 1, device=self.device).view(gw2 - 2, 1).div(2, rounding_mode="trunc")
            i_idxs = torch.tensor([[0, 1, 0, 1], [-1, 0, -1, 0]] * ((gw2 - 2) // 2), device=self.device)
            self.i_idxs = i_idxs + i_floor
        else:
            self.i_idxs = torch.empty(0, 4, dtype=torch.long, device=self.device)

        if gh2 > 2 and gw2 > 2:
            self._j_idxs_rep = self.j_idxs.repeat(max(gh2 - 2, 1), 1, 1).permute(1, 0, 2)
            self._i_idxs_rep = self.i_idxs.repeat(max(gw2 - 2, 1), 1, 1)
        else:
            self._j_idxs_rep = None
            self._i_idxs_rep = None

    def _compute_tiles_inplace(self, img: torch.Tensor) -> None:
        batch = img
        if self.pad_vert > 0 or self.pad_horz > 0:
            batch = F.pad(batch, [0, self.pad_horz, 0, self.pad_vert], mode="reflect")
        self.img_padded.copy_(batch)

        c = self.img_padded.shape[-3]
        tiles = (
            self.img_padded.unfold(1, c, c)
            .unfold(2, self.kernel_vert, self.kernel_vert)
            .unfold(3, self.kernel_horz, self.kernel_horz)
            .squeeze(1)
        )
        self.hist_tiles.copy_(tiles)

    def _compute_interp_tiles_inplace(self) -> None:
        c = self.img_padded.shape[-3]
        interp = (
            self.img_padded.unfold(1, c, c)
            .unfold(2, self.th2, self.th2)
            .unfold(3, self.tw2, self.tw2)
            .squeeze(1)
        )
        self.interp_tiles.copy_(interp)

    def _compute_luts_inplace(self, clip_limit: float) -> None:
        num_bins = 256
        pixels = self.kernel_vert * self.kernel_horz

        tiles = self.hist_tiles.view(self.num_tiles, pixels)
        self.tiles_flat.copy_(tiles)

        self.hist_indices_f.copy_(self.tiles_flat).mul_(num_bins)
        self.hist_indices.copy_(self.hist_indices_f).clamp_(0, num_bins - 1)
        self.histos.zero_()
        self.histos.scatter_add_(1, self.hist_indices, self.ones_flat)

        histos = self.histos
        if clip_limit > 0.0:
            max_val = max(clip_limit * pixels // num_bins, 1)
            histos.clamp_(max=max_val)
            clipped = pixels - histos.sum(1)
            residual = torch.remainder(clipped, num_bins)
            redist = (clipped - residual).div(num_bins)
            histos += redist.unsqueeze(1)
            histos += (self.v_range.unsqueeze(0) < residual.unsqueeze(1)).to(histos.dtype)

        lut_scale = (num_bins - 1) / pixels
        torch.cumsum(histos, 1, out=self.luts_flat_buf)
        self.luts_flat_buf.mul_(lut_scale).clamp_(0, num_bins - 1).floor_()
        self.luts.copy_(self.luts_flat_buf.view(1, self.gh, self.gw, 1, num_bins))

    def _map_luts_inplace(self) -> None:
        luts = self.luts
        gh2, gw2 = self.gh2, self.gw2

        self.mapped_luts.fill_(-1)

        self.mapped_luts[:, 0 :: gh2 - 1, 0 :: gw2 - 1, 0] = luts[
            :, 0 :: max(gh2 // 2 - 1, 1), 0 :: max(gw2 // 2 - 1, 1)
        ]

        if gh2 > 2:
            self.mapped_luts[:, 1:-1, 0 :: gw2 - 1, 0] = luts[:, self.j_idxs[:, 0], 0 :: max(gw2 // 2 - 1, 1)]
            self.mapped_luts[:, 1:-1, 0 :: gw2 - 1, 1] = luts[:, self.j_idxs[:, 2], 0 :: max(gw2 // 2 - 1, 1)]

        if gw2 > 2:
            self.mapped_luts[:, 0 :: gh2 - 1, 1:-1, 0] = luts[:, 0 :: max(gh2 // 2 - 1, 1), self.i_idxs[:, 0]]
            self.mapped_luts[:, 0 :: gh2 - 1, 1:-1, 1] = luts[:, 0 :: max(gh2 // 2 - 1, 1), self.i_idxs[:, 1]]

        if gh2 > 2 and gw2 > 2:
            self.mapped_luts[:, 1:-1, 1:-1, :] = luts[:, self._j_idxs_rep, self._i_idxs_rep]

    def _equalize_tiles_inplace(self) -> None:
        interp = self.interp_tiles
        mapped_luts = self.mapped_luts
        gh2, gw2 = self.gh2, self.gw2
        th, tw = self.th2, self.tw2
        thw = th * tw

        torch.mul(interp, 255, out=self._flatten_f)
        self._flatten_i.copy_(self._flatten_f.flatten(-2, -1))
        self.flatten_indices.copy_(self._flatten_i.unsqueeze(-3).expand(1, gh2, gw2, 4, 1, thw))
        torch.gather(mapped_luts, 5, self.flatten_indices,
                     out=self.preinterp_tiles.view(1, gh2, gw2, 4, 1, thw))

        self.tiles_equalized.zero_()
        preinterp = self.preinterp_tiles

        tl, tr, bl, br = preinterp[:, 1:-1, 1:-1].unbind(3)
        self._t_buf.copy_(tl).sub_(tr).mul_(self.tiw).add_(tr)
        self._b_buf.copy_(bl).sub_(br).mul_(self.tiw).add_(br)
        self._sub_buf.copy_(self._t_buf).sub_(self._b_buf).mul_(self.tih).add_(self._b_buf)
        self.tiles_equalized[:, 1:-1, 1:-1].copy_(self._sub_buf)

        self.tiles_equalized[:, 0 :: gh2 - 1, 0 :: gw2 - 1] = preinterp[:, 0 :: gh2 - 1, 0 :: gw2 - 1, 0]

        top, bottom, _, _ = preinterp[:, 1:-1, 0].unbind(2)
        self._edge_buf.copy_(top).sub_(bottom).mul_(self.tih.squeeze(1)).add_(bottom)
        self.tiles_equalized[:, 1:-1, 0].copy_(self._edge_buf)

        top, bottom, _, _ = preinterp[:, 1:-1, gh2 - 1].unbind(2)
        self._edge_buf.copy_(top).sub_(bottom).mul_(self.tih.squeeze(1)).add_(bottom)
        self.tiles_equalized[:, 1:-1, gh2 - 1].copy_(self._edge_buf)

        left, right, _, _ = preinterp[:, 0, 1:-1].unbind(2)
        self._edge_buf.copy_(left).sub_(right).mul_(self.tiw).add_(right)
        self.tiles_equalized[:, 0, 1:-1].copy_(self._edge_buf)

        left, right, _, _ = preinterp[:, gw2 - 1, 1:-1].unbind(2)
        self._edge_buf.copy_(left).sub_(right).mul_(self.tiw).add_(right)
        self.tiles_equalized[:, gw2 - 1, 1:-1].copy_(self._edge_buf)

        self.tiles_equalized.div_(255.0)

    def _reconstruct_inplace(self) -> None:
        self._perm_buf.copy_(self.tiles_equalized.permute(0, 3, 1, 4, 2, 5))
        self.eq_output.copy_(self._perm_buf.view(1, 1, self.padded_h, self.padded_w))

    @torch.no_grad()
    def apply(self, y_channel: torch.Tensor, out: torch.Tensor, clip_limit: float = 4.0) -> None:
        """Apply CLAHE on Y channel (1,1,H,W) and write result to out."""
        self._compute_tiles_inplace(y_channel)
        self._compute_interp_tiles_inplace()
        self._compute_luts_inplace(clip_limit)
        self._map_luts_inplace()
        self._equalize_tiles_inplace()
        self._reconstruct_inplace()
        out.copy_(self.eq_output[..., : self.height, : self.width])


def rgb_to_yuv_inplace(rgb: torch.Tensor, yuv_out: torch.Tensor, kernels: _ColorKernels) -> None:
    """RGB (1,3,H,W) -> YUV in-place via pre-registered conv weights."""
    flat_in = rgb.reshape(-1, 3, rgb.shape[-2], rgb.shape[-1])
    flat_out = yuv_out.reshape(-1, 3, yuv_out.shape[-2], yuv_out.shape[-1])
    flat_out.copy_(F.conv2d(flat_in, kernels.rgb2yuv_weight))


def yuv_to_rgb_inplace(yuv: torch.Tensor, rgb_out: torch.Tensor, kernels: _ColorKernels) -> None:
    """YUV (1,3,H,W) -> RGB in-place via pre-registered conv weights."""
    flat_in = yuv.reshape(-1, 3, yuv.shape[-2], yuv.shape[-1])
    flat_out = rgb_out.reshape(-1, 3, rgb_out.shape[-2], rgb_out.shape[-1])
    flat_out.copy_(F.conv2d(flat_in, kernels.yuv2rgb_weight))


@torch.no_grad()
def apply_clahe_korina(
    img_tensor: torch.Tensor,
    yuv_buffer: torch.Tensor,
    y_enhanced_buffer: torch.Tensor,
    rgb_out: torch.Tensor,
    clahe_workspace: ClaheWorkspace,
    color_kernels: _ColorKernels,
    clip_limit: float = 4.0,
) -> None:
    """
    對 16-bit RGB 影像進行 CLAHE 增強，全程使用預配置 buffer，避免每幀 CUDA 配置。
    """
    if img_tensor.dtype != torch.float32:
        raise ValueError("輸入必須是 torch.float32 Tensor")

    rgb_to_yuv_inplace(img_tensor, yuv_buffer, color_kernels)
    clahe_workspace.apply(yuv_buffer[:, 0:1, :, :], y_enhanced_buffer, clip_limit=clip_limit)
    yuv_buffer[:, 0:1, :, :].copy_(y_enhanced_buffer)
    yuv_to_rgb_inplace(yuv_buffer, img_tensor, color_kernels)

    out_rgb = img_tensor.squeeze(0).permute(1, 2, 0)
    rgb_out.copy_(out_rgb)
