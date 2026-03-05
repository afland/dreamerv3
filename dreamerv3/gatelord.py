import jax
import jax.numpy as jnp
import ninjax as nj
import embodied.jax.nets as nn

f32 = jnp.float32
sg = jax.lax.stop_gradient


class GateLord(nj.Module):
  """Paper Eq. 24-26, headless (no output gate).

  Per-dimension vector gate with ReTanh activation and additive noise.
  Coupled gate: same gate controls both boundary detection and update magnitude.
  Context update: c_t = gate * c_hat + (1 - gate) * c_{t-1}
  """

  noise_scale: float = 0.1
  act: str = 'tanh'
  norm: str = 'rms'

  def __init__(self, units, **kw):
    self.units = units
    self.kw = kw

  def initial(self, bsize):
    return jnp.zeros([bsize, self.units], nn.COMPUTE_DTYPE)

  def __call__(self, inputs, context, training):
    # inputs: [B, D] (pre-projected shared stoch_proj + action_proj)
    # context: [B, m]
    # Returns: (new_context [B, m], gates [B, m])
    x = jnp.concatenate([inputs, context], -1)

    # Eq. 24: candidate proposal
    c_hat = self.sub('proposal', nn.Linear, self.units, **self.kw)(x)
    c_hat = nn.act(self.act)(self.sub('proposal_norm', nn.Norm, self.norm)(c_hat))

    # Eq. 25: update gate with noise
    gate_pre = self.sub('gate', nn.Linear, self.units, **self.kw)(x)
    if training and self.noise_scale > 0:
      noise = jax.random.normal(nj.seed(), gate_pre.shape, gate_pre.dtype)
      gate_pre = gate_pre + noise * self.noise_scale
    gate = jax.nn.relu(jnp.tanh(gate_pre))  # ReTanh

    # Eq. 26: context update
    c_new = gate * c_hat + (1 - gate) * context
    return c_new, gate


class TimeLord(nj.Module):
  """Decoupled boundary/magnitude gate cell.

  Three separate heads: hard binary boundary gate, soft magnitude scale,
  and candidate update. The boundary gate is binarized with straight-through
  heaviside in the forward pass, so sparsity pressure on the gate does not
  affect update magnitudes.

  Context update: c_t = c_{t-1} + gate * scale * (update - c_{t-1})
  where gate is hard 0/1 (STE) and scale is soft sigmoid.
  """

  noise_scale: float = 0.1
  norm: str = 'rms'

  def __init__(self, units, **kw):
    self.units = units
    self.kw = kw

  def initial(self, bsize):
    return jnp.zeros([bsize, self.units], nn.COMPUTE_DTYPE)

  def __call__(self, inputs, context, training):
    # inputs: [B, D] (pre-projected shared stoch_proj + action_proj)
    # context: [B, m]
    # Returns: (new_context [B, m], gates [B, m])
    x = jnp.concatenate([inputs, context], -1)

    # Three separate heads
    update = self.sub('update', nn.Linear, self.units, **self.kw)(x)
    update = jnp.tanh(update)  # candidate value [-1, 1]

    scale = self.sub('scale', nn.Linear, self.units, **self.kw)(x)
    scale = jax.nn.sigmoid(scale)  # soft magnitude [0, 1]

    gate_pre = self.sub('gate', nn.Linear, self.units, **self.kw)(x)
    if training and self.noise_scale > 0:
      noise = jax.random.normal(nj.seed(), gate_pre.shape, gate_pre.dtype)
      gate_pre = gate_pre + noise * self.noise_scale
    gate_soft = jax.nn.silu(gate_pre)
    # Hard binary boundary via straight-through heaviside
    gate = sg(f32(gate_soft > 0) - gate_soft) + gate_soft
    gate = nn.cast(gate)

    # Context update: only updates when gate = 1
    c_new = context + gate * scale * (update - context)
    return c_new, gate


def sparsity_loss(gates, free=0.0, mode='mean'):
  """Compute gate sparsity loss.

  Args:
    gates: [B, T, m] gate activations over a sequence.
    free: threshold below which sparsity is not penalized.
    mode: 'mean' for smooth mean gate activation per timestep (for GateLord).
          'budget' for binary count of openings with a free budget per
          sequence (for TimeLord).

  Returns:
    [B, T] sparsity loss per timestep.
  """
  if mode == 'mean':
    # Mean gate activation across dims per timestep
    activity = gates.mean(-1)  # [B, T]
    if free > 0:
      activity = jnp.maximum(activity - free, 0.0)
    return activity
  elif mode == 'budget':
    # Any gate > 0 counts as an opening (already binary for TimeLord)
    hard = f32(gates > 0)
    activity = hard.max(-1)  # [B, T] - any dim open = 1
    # Sum over time, apply free budget, normalize back to per-timestep
    T = gates.shape[1]
    total = activity.sum(-1)  # [B]
    if free > 0:
      total = jnp.maximum(total - free, 0.0)
    return (total / T)[:, None].repeat(T, -1)  # [B, T]
  else:
    raise NotImplementedError(mode)
