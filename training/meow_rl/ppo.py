"""The clipped PPO update with separate actor and critic optimizers (docs/MEOWMIND_RL_PLAN.md section 4).

Fixed recipe (profiles/meowmind_rl_pilot.json "ppo", plus the choices this file freezes):
  * actor loss  -mean(min(rho A, clip(rho, 1 - eps, 1 + eps) A)) - c_ent * mean(H(pi(. | o)))
    rho = exp(log pi_new(a | o) - log pi_old(a | o)) for the sampled joint a under the stored mask;
  * critic loss  c_v * mean((V(o) - R)^2), no value clipping;
  * Adam per network (betas 0.9/0.999, eps 1e-5, no weight decay), gradient norm clipped per network;
  * advantages normalized once per collected batch (population std, eps 1e-8); ratios, normalization and loss
    reductions in float32;
  * epochs over reshuffled minibatches (the permutation comes from the checkpointed generator);
  * approximate KL  k = mean(rho - 1 - log rho)  of each actor minibatch, measured before its step: when it exceeds
    target_kl the actor takes no further steps in this update (the step is skipped and recorded); the critic finishes
    all epochs.
Non-finite losses or gradients raise: a broken update is a bug, never skipped silently.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .nets import Actor, Critic, entropy

ADAM_EPS = 1e-5
ADV_EPS = 1e-8


@dataclass
class PPOConfig:
    clip: float = 0.1
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    value_loss_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    epochs_per_update: int = 4
    target_kl: float = 0.01
    minibatch: int = 1024
    gamma: float = 1.0
    gae_lambda: float = 0.95

    @classmethod
    def from_profile(cls, ppo: dict) -> "PPOConfig":
        if ppo.get("value_clipping"):
            raise ValueError("value clipping is off in the pilot")
        keys = cls.__dataclass_fields__
        return cls(**{k: ppo[k] for k in keys if k in ppo})

    def record(self) -> dict:
        return asdict(self) | {"adam_eps": ADAM_EPS, "adam_betas": [0.9, 0.999], "weight_decay": 0.0,
                               "advantage_eps": ADV_EPS, "kl_estimator": "mean(rho-1-log rho), before the step",
                               "kl_stop": "actor stops for the rest of the update; critic continues"}


class Learner:
    def __init__(self, actor: Actor, critic: Critic, cfg: PPOConfig, device: torch.device):
        self.actor, self.critic, self.cfg, self.device = actor.to(device), critic.to(device), cfg, device
        self.opt_actor = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr, betas=(0.9, 0.999), eps=ADAM_EPS)
        self.opt_critic = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr, betas=(0.9, 0.999), eps=ADAM_EPS)

    def state_dict(self) -> dict:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "opt_actor": self.opt_actor.state_dict(), "opt_critic": self.opt_critic.state_dict()}

    def load_state_dict(self, s: dict):
        self.actor.load_state_dict(s["actor"])
        self.critic.load_state_dict(s["critic"])
        self.opt_actor.load_state_dict(s["opt_actor"])
        self.opt_critic.load_state_dict(s["opt_critic"])

    @torch.no_grad()
    def values(self, value_feats: np.ndarray, chunk: int = 8192) -> np.ndarray:
        out = []
        for i in range(0, len(value_feats), chunk):
            x = torch.from_numpy(np.ascontiguousarray(value_feats[i:i + chunk])).to(self.device)
            out.append(self.critic(x).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    @torch.no_grad()
    def log_probs(self, feats: np.ndarray, mask: np.ndarray, chunk: int = 4096) -> np.ndarray:
        out = []
        for i in range(0, len(feats), chunk):
            f = torch.from_numpy(np.ascontiguousarray(feats[i:i + chunk])).to(self.device)
            m = torch.from_numpy(np.ascontiguousarray(mask[i:i + chunk])).to(self.device)
            out.append(self.actor.log_probs(f, m).cpu().numpy())
        return np.concatenate(out)

    def update(self, batch: dict, rng: np.random.Generator) -> dict:
        """One PPO update on a collected batch: feats [B,3,5,112], mask [B,125] bool, action [B], old_logp [B],
        value_feats [B,416], adv [B], ret [B] (numpy). Returns diagnostics."""
        cfg, dev = self.cfg, self.device
        B = len(batch["action"])
        if B == 0:
            raise ValueError("empty batch")
        t = {k: torch.from_numpy(np.ascontiguousarray(batch[k])).to(dev)
             for k in ("feats", "mask", "action", "old_logp", "value_feats", "adv", "ret")}
        t["action"] = t["action"].long()
        adv = t["adv"].float()
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + ADV_EPS)
        stats = {"pg_loss": [], "v_loss": [], "entropy": [], "approx_kl": [], "clip_frac": [],
                 "actor_grad_norm": [], "critic_grad_norm": []}
        actor_steps = critic_steps = 0
        kl_stop = None
        for epoch in range(cfg.epochs_per_update):
            perm = torch.from_numpy(rng.permutation(B)).to(dev)
            for start in range(0, B, cfg.minibatch):
                idx = perm[start:start + cfg.minibatch]
                if kl_stop is None:
                    logp_all = self.actor.log_probs(t["feats"][idx], t["mask"][idx])
                    new = logp_all.gather(1, t["action"][idx, None])[:, 0]
                    if not torch.isfinite(new).all():
                        raise FloatingPointError("a stored action has zero probability under its stored mask")
                    d = new - t["old_logp"][idx].float()
                    rho = d.exp()
                    kl = float((rho - 1 - d).mean())
                    if kl > cfg.target_kl:
                        kl_stop = {"epoch": epoch, "minibatch_start": start, "approx_kl": kl}
                    else:
                        a = adv[idx]
                        pg = -torch.minimum(rho * a, rho.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
                        ent = entropy(logp_all, t["mask"][idx]).mean()
                        loss = pg - cfg.entropy_coefficient * ent
                        self.opt_actor.zero_grad(set_to_none=True)
                        loss.backward()
                        gn = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                        if not (torch.isfinite(loss) and torch.isfinite(gn)):
                            raise FloatingPointError(f"non-finite actor loss {float(loss)} or gradient norm {float(gn)}")
                        self.opt_actor.step()
                        actor_steps += 1
                        stats["pg_loss"].append(float(pg))
                        stats["entropy"].append(float(ent))
                        stats["approx_kl"].append(kl)
                        stats["clip_frac"].append(float(((rho - 1).abs() > cfg.clip).float().mean()))
                        stats["actor_grad_norm"].append(float(gn))
                v = self.critic(t["value_feats"][idx])
                vloss = cfg.value_loss_coefficient * ((v - t["ret"][idx]) ** 2).mean()
                self.opt_critic.zero_grad(set_to_none=True)
                vloss.backward()
                gn = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                if not (torch.isfinite(vloss) and torch.isfinite(gn)):
                    raise FloatingPointError(f"non-finite critic loss {float(vloss)} or gradient norm {float(gn)}")
                self.opt_critic.step()
                critic_steps += 1
                stats["v_loss"].append(float(vloss))
                stats["critic_grad_norm"].append(float(gn))
        out = {k: (float(np.mean(v)) if v else None) for k, v in stats.items()}
        with torch.no_grad():
            pred = torch.from_numpy(self.values(batch["value_feats"]))
            ret = torch.from_numpy(np.asarray(batch["ret"], np.float32))
            var = float(ret.var(unbiased=False))
            out["explained_variance_after"] = None if var == 0 else 1 - float((ret - pred).var(unbiased=False)) / var
        out |= {"batch_decisions": B, "actor_steps": actor_steps, "critic_steps": critic_steps, "kl_stop": kl_stop,
                "max_actor_grad_norm": max(stats["actor_grad_norm"], default=None),
                "max_critic_grad_norm": max(stats["critic_grad_norm"], default=None)}
        for k in ("pg_loss", "v_loss"):
            if out[k] is not None and not math.isfinite(out[k]):
                raise FloatingPointError(k)
        return out
