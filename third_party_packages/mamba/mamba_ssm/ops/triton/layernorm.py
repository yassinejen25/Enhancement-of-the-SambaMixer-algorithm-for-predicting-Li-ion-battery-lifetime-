import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, **factory_kwargs))
        self.bias = None  # Original code has bias = None

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        if residual is not None:
            x = x + residual
        # Compute RMS norm
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        out = self.weight * x_norm
        if prenorm:
            return out, x
        return out


# Optional: CPU fallback LayerNorm if you need it too
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, **factory_kwargs))
        self.bias = nn.Parameter(torch.zeros(hidden_size, **factory_kwargs))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean)**2).mean(dim=-1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias
