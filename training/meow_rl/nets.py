"""PyTorch actor and critics for learning (CUDA on the host, CPU in tests).

The actor is the NumPy reference of training.meow_rl.policy in PyTorch: the same weights give the same joint log
probabilities up to float32 summation order (tests/test_meow_rl.py). Plain ReLU, not fusion.torch_backend's clipped
[0, 1] activation. The critics are fresh MLPs on the 416 public value features with ReLU hidden layers and a linear
output in reward units (identity.REWARD_UNITS).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .identity import PF, TEMPERATURE, VF
from .policy import ACTOR_SHAPES, JA, JB, JC, check_actor


def pick_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda: this PyTorch has no CUDA (install a CUDA build, training/specs/ENVIRONMENT.md)")
    return dev


def configure(dev: torch.device, tf32: bool) -> dict:
    """Explicit matmul precision (the plan: FP32 first; TF32 only as a recorded variant)."""
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    info = {"torch": torch.__version__, "device": str(dev), "tf32": tf32}
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(dev)
        info |= {"gpu": p.name, "gpu_memory_gb": round(p.total_memory / 2 ** 30, 2), "cuda": torch.version.cuda}
    return info


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0, self.l1, self.l2 = nn.Linear(PF, 64), nn.Linear(64, 32), nn.Linear(32, 1)
        for name, idx in (("ja", JA), ("jb", JB), ("jc", JC)):
            self.register_buffer(name, torch.as_tensor(idx, dtype=torch.long), persistent=False)

    def load_numpy(self, w: dict):
        w = check_actor(w)
        with torch.no_grad():
            for i, layer in enumerate((self.l0, self.l1, self.l2)):
                layer.weight.copy_(torch.from_numpy(w[f"w{i}"]))
                layer.bias.copy_(torch.from_numpy(w[f"b{i}"]))
        return self

    def numpy(self) -> dict:
        out = {}
        for i, layer in enumerate((self.l0, self.l1, self.l2)):
            out[f"w{i}"] = layer.weight.detach().cpu().numpy().astype(np.float32).copy()
            out[f"b{i}"] = layer.bias.detach().cpu().numpy().astype(np.float32).copy()
        assert {k: v.shape for k, v in out.items()} == ACTOR_SHAPES
        return out

    def unit_scores(self, feats: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.l0(feats))
        z = torch.relu(self.l1(h))
        return self.l2(z)[..., 0]

    def log_probs(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """feats [B, 3, 5, 112], mask [B, 125] bool -> log pi [B, 125] (float32, -inf where infeasible)."""
        s = self.unit_scores(feats) / TEMPERATURE
        z = s[:, 0, self.ja] + s[:, 1, self.jb] + s[:, 2, self.jc]
        z = z.masked_fill(~mask, float("-inf"))
        return torch.log_softmax(z.float(), dim=-1)


def entropy(logp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Entropy of each row over its feasible joints only (no 0 * -inf)."""
    safe = logp.masked_fill(~mask, 0.0)
    return -(safe.exp() * safe).masked_fill(~mask, 0.0).sum(-1)


class Critic(nn.Module):
    def __init__(self, hidden: list[int]):
        super().__init__()
        sizes = [VF, *hidden, 1]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes, sizes[1:]))
        self.hidden = list(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)[..., 0]

    def parameters_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

