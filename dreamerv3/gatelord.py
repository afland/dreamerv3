import jax
import jax.numpy as jnp
import ninjax as nj
import embodied.jax.nets as nn

f32 = jnp.float32
sg = jax.lax.stop_gradient


class GateLord(nj.Module):
  """Per-dimension vector gate with ReTanh activation and output gate.

  Coupled gate: same gate controls both boundary detection and update magnitude.
  Context update: c_t = gate * c_hat + (1 - gate) * c_{t-1}
  Output gate: sigmoid(a) * tanh(b) read-out from [inputs, c_new].
  """

  noise_scale: float = 0.1

  def __init__(self, units, **kw):
    self.units = units
    self.kw = kw

  def initial(self, bsize):
    return jnp.zeros([bsize, self.units], nn.COMPUTE_DTYPE)

  def __call__(self, inputs, context, training):
    # inputs: [B, D] (pre-projected shared stoch_proj + action_proj)
    # context: [B, m]
    # Returns: (output [B, m], new_context [B, m], gates [B, m])
    x = jnp.concatenate([inputs, context], -1)

    # Candidate proposal
    c_hat = jnp.tanh(self.sub('proposal', nn.Linear, self.units, **self.kw)(x))

    # Update gate with noise
    gate_pre = self.sub('gate', nn.Linear, self.units, **self.kw)(x)
    if training and self.noise_scale > 0:
      noise = jax.random.normal(nj.seed(), gate_pre.shape, gate_pre.dtype)
      gate_pre = gate_pre + noise * self.noise_scale
    gate = jax.nn.relu(jnp.tanh(gate_pre))  # ReTanh

    # Context update
    c_new = gate * c_hat + (1 - gate) * context

    # Output gate: learned read-out for downstream heads
    out = self.sub('out', nn.Linear, 2 * self.units, **self.kw)(
        jnp.concatenate([inputs, c_new], -1))
    a, b = jnp.split(out, 2, -1)
    output = jax.nn.sigmoid(a) * jnp.tanh(b)

    return output, c_new, gate


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

    # Output gate: learned read-out for downstream heads
    out = self.sub('out', nn.Linear, 2 * self.units, **self.kw)(
        jnp.concatenate([inputs, c_new], -1))
    a, b = jnp.split(out, 2, -1)
    output = jax.nn.sigmoid(a) * jnp.tanh(b)

    return output, c_new, gate


def _ste_heaviside(x):
  """Heaviside step with straight-through gradient."""
  return sg(f32(x > 0) - x) + x


def sparsity_loss(gates, free=0.0, mode='mean'):
  """Compute gate sparsity loss.

  Both modes sum activity over time, subtract a free budget, then divide
  by T to get a per-timestep-scale loss matching other [B, T] losses.

  Args:
    gates: [B, T, m] gate activations over a sequence.
    free: number of context changes allowed per sequence before penalty.
    mode: 'mean' for smooth mean gate activation per timestep (for GateLord).
          'budget' for binary count of openings with STE (for TimeLord).

  Returns:
    [B, T] sparsity loss per timestep.
  """
  T = gates.shape[1]
  if mode == 'mean':
    # Mean gate activation across dims per timestep
    activity = gates.mean(-1)  # [B, T]
    total = activity.sum(-1)  # [B]
    if free > 0:
      total = jnp.maximum(total - free, 0.0)
    return (total / T)[:, None].repeat(T, -1)  # [B, T]
  elif mode == 'budget':
    # Any gate dim open counts as one opening (STE for gradient flow)
    activity = _ste_heaviside(gates.max(-1))  # [B, T]
    total = activity.sum(-1)  # [B]
    if free > 0:
      total = jnp.maximum(total - free, 0.0)
    return (total / T)[:, None].repeat(T, -1)  # [B, T]
  else:
    raise NotImplementedError(mode)
