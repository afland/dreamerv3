# BSuite Memory Length Experiment

Comparing 5 configurations on `bsuite memory_length` tasks at difficulty levels
where DreamerV3's GRU-based RSSM is expected to drop off (20-30 memory steps).

## Difficulty Levels

| Task ID | Memory Steps | DV3 (R2I paper) | R2I (R2I paper) |
|---------|-------------|-----------------|-----------------|
| `memory_len/0` | 1 | ~1.0 | ~1.0 |
| `memory_len/1` | 2 | ~1.0 | ~1.0 |
| `memory_len/2` | 3 | ~1.0 | ~1.0 |
| `memory_len/3` | 4 | ~1.0 | ~1.0 |
| `memory_len/4` | 5 | ~1.0 | ~1.0 |
| `memory_len/5` | 7 | ~1.0 | ~1.0 |
| `memory_len/6` | 8 | ~1.0 | ~1.0 |
| `memory_len/7` | 9 | ~1.0 | ~1.0 |
| `memory_len/8` | 10 | ~1.0 | ~1.0 |
| `memory_len/9` | 11 | ~1.0 | ~1.0 |
| `memory_len/10` | 12 | ~0.9 | ~1.0 |
| `memory_len/11` | 14 | ~0.8 | ~1.0 |
| `memory_len/12` | 17 | ~0.7 | ~1.0 |
| `memory_len/13` | 20 | ~0.5 | ~1.0 |
| `memory_len/14` | 25 | ~0.4 | ~0.9 |
| `memory_len/15` | 30 | ~0.4 | ~0.8 |

Default commands below use `/13` (20 steps). Change the task ID suffix for other levels.

## R2I Param Counts on BSuite (MLP-only, no CNN)

| Preset | Params |
|--------|--------|
| `xsmall` | 23.8M |
| `small` | 33.7M |
| `medium` | 58.1M |
| `large` | 98.5M |

DV3 `size25m` is ~21.8M on bsuite. R2I `xsmall` (23.8M) is the closest match.

## Methods

| Method | Branch / Repo | Size Preset | Key Difference |
|--------|--------------|-------------|----------------|
| DreamerV3 | `main` | `size25m` | Baseline GRU-RSSM |
| C-RSSM | `mythicksmallmcts` | `size25m` | Context-augmented RSSM with sparse boundary gates |
| C-RSSM + THICK (with coarse critic) | `mythicksmallmcts` | `size25m` | + hierarchical planning with coarse value bootstrap |
| C-RSSM + THICK (no coarse critic) | `mythicksmallmcts` | `size25m` | + hierarchical planning, no leaf value bootstrap |
| HAUX Bootstrap | `bootstrap` | `size25m` | Multi-horizon auxiliary value targets |
| R2I | `../Recall2Imagine` | `xsmall` | SSM-based (S5) world model |

## Shared Settings

All runs use: `batch_length=64`, `batch_size=16`, `train_ratio=512`,
`steps=1e6`, `envs=4`, `jax.prealloc=False`.

## GPU

3090 at $0.13/hr.

---

## Commands

### A) R2I (from `../Recall2Imagine` repo)

```bash
python recall2imagine/train.py --configs bsuite xsmall \
  --task dm_bsuite_memory_len/13 \
  --jax.prealloc False \
  --logdir ./logdir/r2i_memlen13
```

### B) DreamerV3 Baseline (branch: `main`)

```bash
git checkout main

python dreamerv3/main.py --configs bsuite size25m \
  --task bsuite_memory_len/13 \
  --batch_length 64 --batch_size 16 \
  --run.train_ratio 512 --run.steps 1e6 --run.envs 4 \
  --jax.prealloc False \
  --logdir ./logdir/dv3_memlen13
```

### C) C-RSSM (branch: `mythicksmallmcts`)

#### C-RSSM only (no THICK)

```bash
git checkout mythicksmallmcts

python dreamerv3/main.py --configs bsuite size25m \
  --task bsuite_memory_len/13 \
  --batch_length 64 --batch_size 16 \
  --run.train_ratio 512 --run.steps 1e6 --run.envs 4 \
  --jax.prealloc False \
  --agent.dyn.typ crssm \
  --agent.loss_scales.sparse 1.0 \
  --logdir ./logdir/crssm_memlen13
```

#### C-RSSM + THICK with Coarse Critic

```bash
git checkout mythicksmallmcts

python dreamerv3/main.py --configs bsuite size25m \
  --task bsuite_memory_len/13 \
  --batch_length 64 --batch_size 16 \
  --run.train_ratio 512 --run.steps 1e6 --run.envs 4 \
  --jax.prealloc False \
  --agent.dyn.typ crssm \
  --agent.loss_scales.sparse 1.0 \
  --agent.thick.enabled True \
  --agent.thick.use_coarse_critic True \
  --logdir ./logdir/crssm_thick_memlen13
```

#### C-RSSM + THICK without Coarse Critic

```bash
git checkout mythicksmallmcts

python dreamerv3/main.py --configs bsuite size25m \
  --task bsuite_memory_len/13 \
  --batch_length 64 --batch_size 16 \
  --run.train_ratio 512 --run.steps 1e6 --run.envs 4 \
  --jax.prealloc False \
  --agent.dyn.typ crssm \
  --agent.loss_scales.sparse 1.0 \
  --agent.thick.enabled True \
  --agent.thick.use_coarse_critic False \
  --logdir ./logdir/crssm_thick_nocoarse_memlen13
```

### D) HAUX Bootstrap (branch: `bootstrap`)

```bash
git checkout bootstrap

python dreamerv3/main.py --configs bsuite size25m \
  --task bsuite_memory_len/13 \
  --batch_length 64 --batch_size 16 \
  --run.train_ratio 512 --run.steps 1e6 --run.envs 4 \
  --jax.prealloc False \
  --agent.haux.enabled True \
  --agent.loss_scales.haux 0.03 \
  --agent.haux.horizons "[1,2,4,8]" \
  --agent.haux.rho 0.5 \
  --logdir ./logdir/bootstrap_memlen13
```
