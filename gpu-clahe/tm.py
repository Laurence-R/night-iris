import torch


@torch.no_grad()
def tm_linear_inplace(img_float32_tensor, ldr_buffer, float_scratch, exposure=1.0):
    """Linear exposure map into a preallocated uint8 buffer."""
    torch.mul(img_float32_tensor, exposure * 255.0, out=float_scratch)
    float_scratch.clamp_(0.0, 255.0)
    ldr_buffer.copy_(float_scratch.to(torch.uint8))
