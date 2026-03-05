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

## C-RSSM with GateLord (coupled gate)

```bash
# Crafter with C-RSSM, default settings
python dreamerv3/main.py \
  --logdir ~/logdir/crssm_crafter/{timestamp} \
  --configs crafter \
  --agent.dyn.typ crssm

# Increase sparse loss weight to encourage sparser context changes
python dreamerv3/main.py \
  --logdir ~/logdir/crssm_sparse/{timestamp} \
  --configs crafter \
  --agent.dyn.typ crssm \
  --agent.loss_scales.sparse 5.0

# Larger context dimension
python dreamerv3/main.py \
  --logdir ~/logdir/crssm_ctx32/{timestamp} \
  --configs crafter \
  --agent.dyn.typ crssm \
  --agent.dyn.crssm.context 32

# Allow small gate activations without penalty (sparse_free threshold)
python dreamerv3/main.py \
  --logdir ~/logdir/crssm_free/{timestamp} \
  --configs crafter \
  --agent.dyn.typ crssm \
  --agent.loss_scales.sparse 2.0 \
  --agent.dyn.crssm.sparse_free 0.05

# Debug mode
python dreamerv3/main.py \
  --logdir ~/logdir/crssm_debug/{timestamp} \
  --configs crafter debug \
  --agent.dyn.typ crssm
```

## C-RSSM with TimeLord (decoupled gate)

```bash
# TimeLord with budget sparsity (N free context switches per sequence)
python dreamerv3/main.py \
  --logdir ~/logdir/timelord_crafter/{timestamp} \
  --configs crafter \
  --agent.dyn.typ crssm \
  --agent.dyn.crssm.gate_type timelord

# Allow 3 free context switches per sequence before penalizing
python dreamerv3/main.py \
  --logdir ~/logdir/timelord_budget3/{timestamp} \
  --configs crafter \
  --agent.dyn.typ crssm \
  --agent.dyn.crssm.gate_type timelord \
  --agent.dyn.crssm.sparse_free 3.0

# Higher sparse loss to enforce fewer context switches
python dreamerv3/main.py \
  --logdir ~/logdir/timelord_strict/{timestamp} \
  --configs crafter \
  --agent.dyn.typ crssm \
  --agent.dyn.crssm.gate_type timelord \
  --agent.loss_scales.sparse 10.0 \
  --agent.dyn.crssm.sparse_free 2.0
```

## Coarse Prior Capacity

The coarse prior predicts stochastic state from `[prev_stoch, prev_action, context]` without seeing the deterministic state `h_t`. By default it uses 1 MLP layer (`coarse_layers: 1`), making it intentionally weaker than the precise prior (`imglayers: 2`). Increasing to 2 layers gives it more capacity to leverage the context signal.

```bash
# Deeper coarse prior (2 layers to match precise prior depth)
python dreamerv3/main.py \
  --logdir ~/logdir/crssm_coarse2/{timestamp} \
  --configs crafter \
  --agent.dyn.typ crssm \
  --agent.dyn.crssm.coarse_layers 2
```

## THICK (full hierarchical model)

```bash
# C-RSSM + HLWM + coarse critic
python dreamerv3/main.py \
  --logdir ~/logdir/thick_crafter/{timestamp} \
  --configs crafter \
  --agent.dyn.typ crssm \
  --agent.thick.enabled True

# THICK with TimeLord gate
python dreamerv3/main.py \
  --logdir ~/logdir/thick_timelord/{timestamp} \
  --configs crafter \
  --agent.dyn.typ crssm \
  --agent.dyn.crssm.gate_type timelord \
  --agent.thick.enabled True \
  --agent.dyn.crssm.sparse_free 3.0

# Adjust value mixing (more weight on lambda-returns vs V^long)
python dreamerv3/main.py \
  --logdir ~/logdir/thick_psi/{timestamp} \
  --configs crafter \
  --agent.dyn.typ crssm \
  --agent.thick.enabled True \
  --agent.thick.psi 0.7

# Atari with THICK, 50m size
python dreamerv3/main.py \
  --logdir ~/logdir/thick_atari/{timestamp} \
  --configs atari size50m \
  --agent.dyn.typ crssm \
  --agent.thick.enabled True \
  --task atari_breakout
```

## Model Size Presets

Available: `size1m`, `size12m`, `size25m`, `size50m`, `size100m`, `size200m` (default), `size400m`.

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
| `--agent.dyn.crssm.gate_type` | Gate cell (`gatelord` or `timelord`) | `gatelord` |
| `--agent.dyn.crssm.context` | Context vector dimension | `16` |
| `--agent.dyn.crssm.gate_noise_scale` | Gate noise std during training | `0.1` |
| `--agent.dyn.crssm.sparse_free` | Sparsity free threshold/budget | `0.0` |
| `--agent.loss_scales.sparse` | Sparsity loss weight | `1.0` |
| `--agent.thick.enabled` | Enable HLWM + coarse critic | `False` |
| `--agent.thick.psi` | V^lambda vs V^long mixing (1.0 = all lambda) | `0.9` |
| `--agent.dyn.crssm.coarse_layers` | MLP layers for coarse prior | `1` |
| `--agent.thick.hl_act_dim` | High-level action categories | `5` |
