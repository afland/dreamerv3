# Example Training Runs

## Basic Format

```bash
python dreamerv3/main.py \
  --logdir ~/logdir/{timestamp} \
  --configs <presets...> \
  --key.subkey value
```

Config presets from `configs.yaml` are applied in order via `--configs`. Any config value can be overridden with flat `--key.subkey value` flags.

## Standard DreamerV3 (RSSM)

```bash
# Crafter, default 200m size
python dreamerv3/main.py \
  --logdir ~/logdir/crafter/{timestamp} \
  --configs crafter

# Atari Pong, 50m model size
python dreamerv3/main.py \
  --logdir ~/logdir/atari/{timestamp} \
  --configs atari size50m \
  --task atari_pong

# DMC Walker (vision), 12m model size
python dreamerv3/main.py \
  --logdir ~/logdir/dmc/{timestamp} \
  --configs dmc_vision size12m \
  --task dmc_walker_walk

# Debug mode (small model, CPU, fast iterations)
python dreamerv3/main.py \
  --logdir ~/logdir/debug/{timestamp} \
  --configs crafter debug
```

## C-RSSM with BlockGRU Context

The context cell is a BlockGRU (half of deter dim) wrapped with a scalar binary boundary gate. Context never enters the fine pathway. The boundary gate fires sparsely, controlled by a Bernoulli KL prior.

```bash
# Crafter with C-RSSM, default settings
python dreamerv3/main.py \
  --logdir ~/logdir/crssm_crafter/{timestamp} \
  --configs crafter \
  --agent.dyn.typ crssm

# Increase sparse loss weight to encourage sparser context updates
python dreamerv3/main.py \
  --logdir ~/logdir/crssm_sparse/{timestamp} \
  --configs crafter \
  --agent.dyn.typ crssm \
  --agent.loss_scales.sparse 5.0

# Lower boundary prior rate (sparser updates, ~5% of timesteps)
python dreamerv3/main.py \
  --logdir ~/logdir/crssm_sparse5/{timestamp} \
  --configs crafter \
  --agent.dyn.typ crssm \
  --agent.dyn.crssm.boundary_prior 0.05

# Higher boundary prior rate (more frequent updates, ~20% of timesteps)
python dreamerv3/main.py \
  --logdir ~/logdir/crssm_freq/{timestamp} \
  --configs crafter \
  --agent.dyn.typ crssm \
  --agent.dyn.crssm.boundary_prior 0.2

# Atari100k with explicit boundary_prior and sparse scale
CUDA_VISIBLE_DEVICES=0 nohup python dreamerv3/main.py \
  --configs atari100k size12m \
  --agent.dyn.typ crssm \
  --agent.dyn.crssm.boundary_prior 0.1 \
  --agent.loss_scales.sparse 1.0 \
  > crssm_blockgru.log 2>&1 &

# Debug mode
python dreamerv3/main.py \
  --logdir ~/logdir/crssm_debug/{timestamp} \
  --configs crafter debug \
  --agent.dyn.typ crssm
```

## THICK (full hierarchical model)

HLWM predicts context at the next boundary (c_tau). The coarse prior provides z_tau from c_tau. The coarse critic evaluates V^c at [c_tau, z_tau].

```bash
# C-RSSM + HLWM + coarse critic
python dreamerv3/main.py \
  --logdir ~/logdir/thick_crafter/{timestamp} \
  --configs crafter \
  --agent.dyn.typ crssm \
  --agent.thick.enabled True

# Adjust value mixing (more weight on lambda-returns vs V^long)
python dreamerv3/main.py \
  --logdir ~/logdir/thick_psi/{timestamp} \
  --configs crafter \
  --agent.dyn.typ crssm \
  --agent.thick.enabled True \
  --agent.thick.psi 0.7

# Split critics: fine on V^lambda, coarse on V^long, mixed baseline
python dreamerv3/main.py \
  --logdir ~/logdir/thick_split/{timestamp} \
  --configs crafter \
  --agent.dyn.typ crssm \
  --agent.thick.enabled True \
  --agent.thick.split_critics True

# Atari with THICK, 50m size
python dreamerv3/main.py \
  --logdir ~/logdir/thick_atari/{timestamp} \
  --configs atari size50m \
  --agent.dyn.typ crssm \
  --agent.thick.enabled True \
  --task atari_breakout

# Sparser boundaries with THICK
python dreamerv3/main.py \
  --logdir ~/logdir/thick_sparse/{timestamp} \
  --configs crafter \
  --agent.dyn.typ crssm \
  --agent.thick.enabled True \
  --agent.dyn.crssm.boundary_prior 0.05 \
  --agent.loss_scales.sparse 2.0
```

## Model Size Presets

Available: `size1m`, `size12m`, `size25m`, `size50m`, `size100m`, `size200m` (default), `size400m`.

Context dim is always deter/2. Coarse hidden is always hidden/2.

```bash
# Small model for quick experiments
python dreamerv3/main.py \
  --configs crafter size12m \
  --agent.dyn.typ crssm

# Large model
python dreamerv3/main.py \
  --configs crafter size400m \
  --agent.dyn.typ crssm
```

## Environment Presets

| Preset | Task example | Notes |
|--------|-------------|-------|
| `crafter` | `crafter_reward` | Procedural survival |
| `atari` | `atari_pong` | Atari 26 games |
| `atari100k` | `atari100k_pong` | 100k step benchmark |
| `dmc_vision` | `dmc_walker_walk` | DeepMind Control (pixels) |
| `dmc_proprio` | `dmc_walker_walk` | DeepMind Control (state) |
| `dmlab` | `dmlab_explore_goal_locations_small` | DeepMind Lab |
| `minecraft` | `minecraft_diamond` | Minecraft diamond |
| `procgen` | `procgen_coinrun` | Procgen benchmark |

## Key Config Values

| Flag | Description | Default |
|------|-------------|---------|
| `--agent.dyn.typ` | World model type (`rssm` or `crssm`) | `rssm` |
| `--agent.dyn.crssm.context` | Context dim (deter/2 per size preset) | `4096` |
| `--agent.dyn.crssm.coarse_hidden` | Coarse pathway hidden dim (hidden/2) | `512` |
| `--agent.dyn.crssm.boundary_prior` | Expected boundary rate (Bernoulli prior) | `0.1` |
| `--agent.loss_scales.sparse` | Boundary KL loss weight | `1.0` |
| `--agent.thick.enabled` | Enable HLWM + coarse critic | `False` |
| `--agent.thick.psi` | V^lambda vs V^long mixing (1.0 = all lambda) | `0.9` |
| `--agent.thick.split_critics` | Train critics on separate targets (V^λ / V^long) | `False` |
| `--agent.thick.hl_act_dim` | High-level action categories | `5` |
