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

## Environments for Evaluating THICK

These environments test temporal reasoning, memory, and long-horizon planning — the capabilities THICK is designed to improve. All are available in-repo and lightweight enough for 12m.

### BSuite Memory Length

Agent must remember an observation across a variable delay and act on it. Difficulty scales via suffix `/0` (short) to `/5` (long delay). Requires `pip install bsuite`.

```bash
# Easy (short delay)
python dreamerv3/main.py \
  --configs bsuite size12m \
  --task bsuite_memory_len/0 \
  --agent.dyn.typ crssm \
  --agent.thick.enabled True

# Hard (long delay)
python dreamerv3/main.py \
  --configs bsuite size12m \
  --task bsuite_memory_len/5 \
  --agent.dyn.typ crssm \
  --agent.thick.enabled True
```

### BSuite Discounting Chain

Agent chooses between short-term and long-term reward chains. Tests temporal discounting and planning horizon. Difficulty `/0` (short chain) to `/4` (long chain). Requires `pip install bsuite`.

```bash
python dreamerv3/main.py \
  --configs bsuite size12m \
  --task bsuite_discounting_chain/0 \
  --agent.dyn.typ crssm \
  --agent.thick.enabled True
```

### PinPad (Sequence Memory)

Agent must press colored pads in the correct order. Directly tests sequential memory. Difficulty scales with number of pads: `pinpad_three` through `pinpad_eight`.

```bash
# 4 pads (moderate)
python dreamerv3/main.py \
  --task pinpad_four \
  --configs size12m \
  --agent.dyn.typ crssm \
  --agent.thick.enabled True \
  --run.train_ratio 256

# 6 pads (hard)
python dreamerv3/main.py \
  --task pinpad_six \
  --configs size12m \
  --agent.dyn.typ crssm \
  --agent.thick.enabled True \
  --run.train_ratio 256
```

### Crafter

Hierarchical survival: collect wood → craft pickaxe → mine stone → etc. Multi-step achievement structure creates natural subgoal boundaries. Has its own config preset.

```bash
python dreamerv3/main.py \
  --configs crafter size12m \
  --agent.dyn.typ crssm \
  --agent.thick.enabled True
```

### MemoryMaze

Purpose-built long-horizon memory environment. Agent navigates a maze and must remember visited locations. Requires `pip install memory-maze`.

```bash
python dreamerv3/main.py \
  --task memmaze_9x9 \
  --configs size12m \
  --agent.dyn.typ crssm \
  --agent.thick.enabled True \
  --run.train_ratio 256
```

### LocoNav (Ant Maze)

Locomotion + navigation in procedural mazes. Maze size controls difficulty: `_s` (small), `_m` (medium), `_l` (large).

```bash
python dreamerv3/main.py \
  --configs loconav size12m \
  --task loconav_ant_maze_m \
  --agent.dyn.typ crssm \
  --agent.thick.enabled True
```

### Suggested Ablation Protocol

For each environment, compare these three configs to isolate THICK's contribution:

1. **Baseline RSSM**: `--agent.dyn.typ rssm`
2. **C-RSSM only**: `--agent.dyn.typ crssm`
3. **C-RSSM + THICK**: `--agent.dyn.typ crssm --agent.thick.enabled True`

## Key Config Values

| Flag | Description | Default |
|------|-------------|---------|
| `--agent.dyn.typ` | World model type (`rssm` or `crssm`) | `rssm` |
| `--agent.dyn.crssm.context` | Context dim (deter/2 per size preset) | `4096` |
| `--agent.dyn.crssm.coarse_hidden` | Coarse pathway hidden dim (hidden/2) | `512` |
| `--agent.dyn.crssm.boundary_prior` | Expected boundary rate (Bernoulli prior) | `0.1` |
| `--agent.dyn.crssm.stochastic_gate` | Bernoulli sampling + STE (`True`) vs hard threshold (`False`) | `True` |
| `--agent.loss_scales.sparse` | Boundary KL loss weight | `1.0` |
| `--agent.thick.enabled` | Enable HLWM + coarse critic | `False` |
| `--agent.thick.psi` | V^lambda vs V^long mixing (1.0 = all lambda) | `0.9` |
| `--agent.thick.split_critics` | Train critics on separate targets (V^λ / V^long) | `True` |
| `--agent.thick.hl_act_dim` | High-level action categories | `5` |
