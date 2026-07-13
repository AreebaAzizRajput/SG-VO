import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """nn.Linear wrapped with a low-rank adapter.

    Forward: y = W x  +  scale * (B A) x
      W stays frozen; only lora_A and lora_B are updated.
    Initialised so that B A = 0, meaning no change at the start of adaptation.
    """

    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 1.0):
        super().__init__()
        self.original = original
        for p in self.original.parameters():
            p.requires_grad_(False)

        in_f, out_f = original.in_features, original.out_features
        # Create adapters on the SAME device/dtype as the wrapped layer, so
        # injection also works after the model was already moved to GPU.
        dev, dt = original.weight.device, original.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(rank, in_f, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, device=dev, dtype=dt))
        self.scale  = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.original(x) + self.scale * F.linear(F.linear(x, self.lora_A), self.lora_B)


class LoRAConv2d(nn.Module):
    """nn.Conv2d wrapped with a low-rank 1×1 adapter path.

    Forward: y = conv(x)  +  scale * up(down(x))
      where down is Conv2d(in, rank, 1) and up is Conv2d(rank, out, 1).
    Works for any kernel size in the original conv; the adapter always uses 1×1.
    Safe as long as the original conv has stride=1 (all targeted layers do).
    """

    def __init__(self, original: nn.Conv2d, rank: int = 8, alpha: float = 1.0):
        super().__init__()
        self.original = original
        for p in self.original.parameters():
            p.requires_grad_(False)

        in_ch, out_ch = original.in_channels, original.out_channels
        self.lora_down = nn.Conv2d(in_ch, rank, kernel_size=1, bias=False)
        self.lora_up   = nn.Conv2d(rank, out_ch, kernel_size=1, bias=False)
        self.scale     = alpha / rank
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        # Match the wrapped layer's device/dtype (model may already be on GPU).
        self.lora_down.to(original.weight.device, original.weight.dtype)
        self.lora_up.to(original.weight.device, original.weight.dtype)

    def forward(self, x):
        return self.original(x) + self.scale * self.lora_up(self.lora_down(x))


# ── Freezing ──────────────────────────────────────────────────────────────────

def freeze_all(model: nn.Module):
    """Freeze every parameter in the model."""
    for p in model.parameters():
        p.requires_grad_(False)


# ── Injection ─────────────────────────────────────────────────────────────────

def inject_lora_pose_net(pose_net: nn.Module, rank: int = 8, alpha: float = 1.0,
                         targets: str = 'attention'):
    """Replace target layers in PoseResNet with LoRA-wrapped versions.

    targets:
      'attention' — Wq, Wk, Wv and the three projection convs in crossAttention
      'decoder'   — the four convs in PoseDecoder.net
      'both'      — all of the above
    Call freeze_all(pose_net) BEFORE this so that only LoRA params are trainable.
    """
    attn = pose_net.crossAttention

    if targets in ('attention', 'both'):
        attn.Wq         = LoRALinear(attn.Wq,         rank=rank, alpha=alpha)
        attn.Wk         = LoRALinear(attn.Wk,         rank=rank, alpha=alpha)
        attn.Wv         = LoRALinear(attn.Wv,         rank=rank, alpha=alpha)
        attn.proj_in_q  = LoRAConv2d(attn.proj_in_q,  rank=rank, alpha=alpha)
        attn.proj_in_kv = LoRAConv2d(attn.proj_in_kv, rank=rank, alpha=alpha)
        attn.proj_out   = LoRAConv2d(attn.proj_out,   rank=rank, alpha=alpha)

    if targets in ('decoder', 'both'):
        # PoseDecoder.net is a ModuleList: [squeeze_conv, pose0, pose1, pose2]
        net = pose_net.decoder.net
        for i in range(len(net)):
            net[i] = LoRAConv2d(net[i], rank=rank, alpha=alpha)


# ── Parameter and state helpers ───────────────────────────────────────────────

def lora_parameters(model: nn.Module):
    """Yield only the LoRA adapter parameters (all have requires_grad=True)."""
    seen = set()
    for module in model.modules():
        if isinstance(module, LoRALinear):
            for p in [module.lora_A, module.lora_B]:
                if id(p) not in seen:
                    seen.add(id(p))
                    yield p
        elif isinstance(module, LoRAConv2d):
            for p in list(module.lora_down.parameters()) + list(module.lora_up.parameters()):
                if id(p) not in seen:
                    seen.add(id(p))
                    yield p


def lora_state_dict(model: nn.Module) -> dict:
    """Return a compact dict containing only the LoRA adapter tensors (cloned).
    Much smaller than a full state_dict — tens of KB vs. ~150 MB."""
    sd = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            sd[f"{name}.lora_A"] = module.lora_A.data.clone()
            sd[f"{name}.lora_B"] = module.lora_B.data.clone()
        elif isinstance(module, LoRAConv2d):
            sd[f"{name}.lora_down"] = module.lora_down.weight.data.clone()
            sd[f"{name}.lora_up"]   = module.lora_up.weight.data.clone()
    return sd


def load_lora_state_dict(model: nn.Module, sd: dict):
    """Load a LoRA state dict (from lora_state_dict) back into the model."""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.lora_A.data.copy_(sd[f"{name}.lora_A"])
            module.lora_B.data.copy_(sd[f"{name}.lora_B"])
        elif isinstance(module, LoRAConv2d):
            module.lora_down.weight.data.copy_(sd[f"{name}.lora_down"])
            module.lora_up.weight.data.copy_(sd[f"{name}.lora_up"])


def count_lora_params(model: nn.Module) -> int:
    return sum(p.numel() for p in lora_parameters(model))
