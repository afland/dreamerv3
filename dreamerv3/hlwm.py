import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

f32 = jnp.float32
sg = jax.lax.stop_gradient


def generate_targets(feat, actions, rewards, discount):
  """Generate HLWM targets from context change points.

  For each timestep t, find the next context change point tau(t) and compute:
  - Target context at tau (the context state at the change point)
  - Time delta = tau(t) - t
  - Accumulated discounted reward from t to tau(t)

  Args:
    feat: dict with 'context' [B,T,m], 'stoch' [B,T,S,C], 'gate_prob' [B,T]
    actions: [B,T,A] raw actions
    rewards: [B,T] rewards
    discount: scalar discount factor

  Returns:
    targets: dict of [B,T,...] target tensors
    valid: [B,T] boolean mask (False where no future change in sequence)
  """
  context = feat['context']  # [B, T, m]
  stoch = feat['stoch']      # [B, T, S, C]
  gate_binary = feat['gate_binary']  # [B, T]
  B, T, m = context.shape

  # Detect context changes: actual gate samples from forward pass
  changes = (gate_binary > 0.5)  # [B, T]

  # For each t, find tau(t) = next change point after t
  def reverse_scan_tau(carry, inp):
    change_here = inp  # [B] bool
    carry = jnp.where(change_here, jnp.zeros_like(carry), carry + 1)
    return carry, carry

  init_carry = jnp.full((B,), T, dtype=jnp.int32)
  _, tau_offsets = jax.lax.scan(
      reverse_scan_tau, init_carry,
      jnp.moveaxis(changes, 1, 0),
      reverse=True)
  tau_offsets = jnp.moveaxis(tau_offsets, 0, 1)  # [B, T]

  t_indices = jnp.arange(T)[None, :]  # [1, T]
  tau_abs = t_indices + tau_offsets  # [B, T]
  valid = (tau_offsets > 0) & (tau_abs < T)  # [B, T]
  tau_abs = jnp.minimum(tau_abs, T - 1)

  # Gather context at tau (the change point)
  b_idx = jnp.arange(B)[:, None].repeat(T, 1)  # [B, T]
  target_context = context[b_idx, tau_abs]  # [B, T, m]

  # Time delta
  time_delta = f32(tau_abs - t_indices)  # [B, T]

  # Accumulated discounted reward
  def compute_inter_reward(carry, inp):
    agg = carry
    rew_t, disc_t = inp
    agg = rew_t + disc_t * agg
    return agg, agg

  change_disc = (1.0 - f32(changes)) * discount
  _, inter_rewards = jax.lax.scan(
      compute_inter_reward,
      jnp.zeros(B),
      (jnp.moveaxis(rewards, 1, 0), jnp.moveaxis(change_disc, 1, 0)),
      reverse=True)
  inter_rewards = jnp.moveaxis(inter_rewards, 0, 1)  # [B, T]

  targets = dict(
      context=target_context,
      time_delta=time_delta,
      inter_reward=inter_rewards,
  )
  return targets, valid


class HLWM(nj.Module):
  """High-Level World Model.

  Predicts transitions between context change points. HLWM predicts
  c_tau (context at next boundary) instead of z and action separately.
  """

  hl_act_dim: int = 5
  layers: int = 3
  units: int = 256
  act: str = 'silu'
  norm: str = 'rms'

  def __init__(self, stoch, classes, context, act_space, **kw):
    self.stoch = stoch
    self.classes = classes
    self.context_dim = context
    self.act_space = act_space
    self.kw = kw
    self.coarse_dim = context + stoch * classes  # [c_t, flatten(z_t)]

  def _body(self, name, x):
    """Shared MLP body."""
    for i in range(self.layers):
      x = self.sub(f'{name}{i}', nn.Linear, self.units, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'{name}{i}norm', nn.Norm, self.norm)(x))
    return x

  def _posterior(self, context_t, stoch_t, context_tau, stoch_tau):
    """Posterior Q_theta: [c_t, z_t, c_tau, z_tau] -> A_t logits."""
    stoch_t_flat = stoch_t.reshape((*stoch_t.shape[:-2], -1))
    stoch_tau_flat = stoch_tau.reshape((*stoch_tau.shape[:-2], -1))
    x = jnp.concatenate([context_t, stoch_t_flat, context_tau, stoch_tau_flat], -1)
    x = self._body('post', x)
    logit = self.sub('post_logit', nn.Linear, self.hl_act_dim, **self.kw)(x)
    return logit

  def _prior(self, context_t, stoch_t):
    """Prior P_theta: [c_t, z_t] -> A_hat_t logits."""
    stoch_t_flat = stoch_t.reshape((*stoch_t.shape[:-2], -1))
    x = jnp.concatenate([context_t, stoch_t_flat], -1)
    x = self._body('prior', x)
    logit = self.sub('prior_logit', nn.Linear, self.hl_act_dim, **self.kw)(x)
    return logit

  def _predict(self, name, hl_act, context_t, stoch_t):
    """Prediction head from [A_t, c_t, z_t]."""
    stoch_t_flat = stoch_t.reshape((*stoch_t.shape[:-2], -1))
    x = jnp.concatenate([hl_act, context_t, stoch_t_flat], -1)
    return self._body(name, x)

  def loss(self, feat, actions, rewards, discount, training):
    """Compute HLWM losses.

    Args:
      feat: dict with context, stoch, gate_prob from C-RSSM observe
      actions: dict of action tensors
      rewards: [B,T] rewards
      discount: scalar
      training: bool

    Returns:
      losses: dict of [B,T] loss tensors
      metrics: dict of scalar metrics
    """
    losses = {}
    metrics = {}

    act_emb = nn.DictConcat(self.act_space, 1)(actions)
    targets, valid = generate_targets(feat, act_emb, rewards, discount)
    valid_f = f32(valid)

    B, T = valid.shape
    context_t = sg(feat['context'])
    stoch_t = sg(feat['stoch'])

    # For posterior, we need stoch at tau. Use the stoch from feat at tau indices.
    # We gather stoch at tau from the same reverse-scan logic in generate_targets.
    gate_binary = feat['gate_binary']
    changes = (gate_binary > 0.5)
    def reverse_scan_tau(carry, inp):
      change_here = inp
      carry = jnp.where(change_here, jnp.zeros_like(carry), carry + 1)
      return carry, carry
    init_carry = jnp.full((B,), T, dtype=jnp.int32)
    _, tau_offsets = jax.lax.scan(
        reverse_scan_tau, init_carry,
        jnp.moveaxis(changes, 1, 0), reverse=True)
    tau_offsets = jnp.moveaxis(tau_offsets, 0, 1)
    t_indices = jnp.arange(T)[None, :]
    tau_abs = jnp.minimum(t_indices + tau_offsets, T - 1)
    b_idx = jnp.arange(B)[:, None].repeat(T, 1)
    stoch_tau = sg(feat['stoch'][b_idx, tau_abs])

    # Posterior and prior HL actions
    post_logit = self._posterior(
        context_t, stoch_t, sg(targets['context']), stoch_tau)
    prior_logit = self._prior(context_t, stoch_t)

    post_dist = embodied.jax.outs.OneHot(post_logit, 0.01)
    prior_dist = embodied.jax.outs.OneHot(prior_logit, 0.01)
    hl_act = nn.cast(post_dist.sample(seed=nj.seed()))

    # Prediction heads from [A_t, c_t, z_t]
    pred_feat = self._predict('pred', hl_act, context_t, stoch_t)

    # Context prediction: predict c_tau (MSE)
    context_pred = self.sub(
        'context_out', nn.Linear, self.context_dim, **self.kw)(pred_feat)
    losses['hlwm_context'] = jnp.square(
        context_pred - sg(targets['context'])).sum(-1) * valid_f

    # Time prediction
    time_pred = self.sub('time_out', nn.Linear, 1, **self.kw)(pred_feat)
    time_pred = time_pred.squeeze(-1)
    losses['hlwm_time'] = jnp.square(
        time_pred - sg(targets['time_delta'])) * valid_f

    # Reward prediction
    rew_pred = self.sub('rew_out', nn.Linear, 1, **self.kw)(pred_feat)
    rew_pred = rew_pred.squeeze(-1)
    losses['hlwm_reward'] = jnp.square(
        rew_pred - sg(targets['inter_reward'])) * valid_f

    # HL action KL: KL(sg(Q) || P) — only train prior toward posterior
    sg_post_dist = embodied.jax.outs.OneHot(sg(post_logit), 0.01)
    act_kl = sg_post_dist.kl(prior_dist)
    losses['hlwm_act_kl'] = act_kl * valid_f

    metrics['hlwm_valid_frac'] = valid_f.mean()
    metrics['hlwm_time_pred'] = time_pred.mean()
    metrics['hlwm_prior_ent'] = prior_dist.entropy().mean()
    metrics['hlwm_post_ent'] = post_dist.entropy().mean()

    return losses, metrics

  def predict(self, context, stoch, training):
    """Sample from prior and predict outcomes for V^long.

    Args:
      context: [B, m] current context
      stoch: [B, S, C] current stochastic state

    Returns:
      dict with predicted context_tau, reward, time_delta
    """
    prior_logit = self._prior(context, stoch)
    prior_dist = embodied.jax.outs.OneHot(prior_logit, 0.01)
    hl_act = nn.cast(prior_dist.sample(seed=nj.seed()))

    pred_feat = self._predict('pred', hl_act, context, stoch)

    # Context prediction
    context_pred = self.sub(
        'context_out', nn.Linear, self.context_dim, **self.kw)(pred_feat)

    time_pred = self.sub('time_out', nn.Linear, 1, **self.kw)(pred_feat)
    time_pred = jax.nn.relu(time_pred.squeeze(-1)) + 1  # at least 1 step

    rew_pred = self.sub('rew_out', nn.Linear, 1, **self.kw)(pred_feat)
    rew_pred = rew_pred.squeeze(-1)

    return dict(
        context=context_pred,
        time_delta=time_pred,
        reward=rew_pred,
    )
