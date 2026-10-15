"""Meowmind-inspired on-policy PPO with a large value critic (task T0054, docs/MEOWMIND_RL_PLAN.md).

Modules:
  identity  architecture/feature/reward-unit ids, the pilot profile and the source identities every run records;
  bridge    ctypes wrapper of tools/cpp/meow_rl/bridge.cpp: the unmodified Meowmind reference's features, feasibility,
            opponent belief and frozen resolver, one session per episode;
  policy    the masked joint actor (NumPy reference for CPU rollouts, PyTorch for learning) and the critics;
  gae       terminal rewards, pairwise utility and generalized advantage estimation;
  env       the live episode adapter: official engine, opponent processes, the learner seat in-process;
  pool      opponent pool, family caps and the episode schedule;
  store     atomic rollout/episode/checkpoint storage with exact resume;
  ppo       the clipped PPO update with separate actor and critic optimizers;
  quant     the integer CPU export of the actor (int8 weights, int16 activations, exact NumPy emulation);
  train     the host entry point (smoke and full-budget modes).
"""
