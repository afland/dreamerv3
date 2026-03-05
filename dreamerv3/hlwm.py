import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

f32 = jnp.float32
sg = jax.lax.stop_gradient


def generate_targets(feat, actions, rewards, discount):
  """Generate HLWM targets from context change points (Eq. 10-11).

  For each timestep t, find the next context change point tau(t) and compute:
  - Target stoch at tau-1 (state just before the change)
  - Target action at tau-1
  - Time delta = tau(t) - t
  - Accumulated discounted reward from t to tau(t)

  Args:
    feat: dict with 'context' [B,T,m], 'stoch' [B,T,S,C], 'gates' [B,T,m]
    actions: [B,T,A] raw actions
    rewards: [B,T] rewards
    discount: scalar discount factor

  Returns:
    targets: dict of [B,T,...] target tensors
    valid: [B,T] boolean mask (False where no future change in sequence)
  """
  context = feat['context']  # [B, T, m]
  stoch = feat['stoch']      # [B, T, S, C]
  gates = feat['gates']      # [B, T, m]
  B, T, m = context.shape

  # Detect context changes: any gate > 0 at timestep t
  changes = (gates.sum(-1) > 0)  # [B, T]

  # For each t, find tau(t) = next change point after t
  # We reverse-scan to propagate the "next change" index backwards
  # tau_idx[b,t] = index of next change point >= t+1, or T if none
  def reverse_scan_tau(carry, inp):
    # carry: [B] next change index
    change_here = inp  # [B] bool
    carry = jnp.where(change_here, jnp.zeros_like(carry), carry + 1)
    return carry, carry

  # Scan backwards over time dimension
  init_carry = jnp.full((B,), T, dtype=jnp.int32)  # no change found yet
  _, tau_offsets = jax.lax.scan(
      reverse_scan_tau, init_carry,
      jnp.moveaxis(changes, 1, 0),  # [T, B]
      reverse=True)
  # tau_offsets[t, b] = how many steps until next change (including t itself)
  tau_offsets = jnp.moveaxis(tau_offsets, 0, 1)  # [B, T]

  # tau(t) = t + tau_offsets[t] (absolute index of next change)
  t_indices = jnp.arange(T)[None, :]  # [1, T]
  tau_abs = t_indices + tau_offsets  # [B, T]
  tau_abs = jnp.minimum(tau_abs, T - 1)  # clamp to valid range

  # Valid mask: tau must be within sequence and > t
  valid = (tau_offsets > 0) & (tau_abs < T)  # [B, T]

  # Gather targets at tau-1 (state just before change)
  # For stoch and action targets, use tau-1; for time delta, use tau-t
  tau_prev = jnp.maximum(tau_abs - 1, 0)  # [B, T]

  # Gather stoch at tau_prev
  b_idx = jnp.arange(B)[:, None].repeat(T, 1)  # [B, T]
  target_stoch = stoch[b_idx, tau_prev]  # [B, T, S, C]
  target_action = actions[b_idx, tau_prev]  # [B, T, A]
  target_context = context[b_idx, tau_abs]  # [B, T, m] context at change point

  # Time delta
  time_delta = f32(tau_abs - t_indices)  # [B, T]

  # Accumulated discounted reward: r_{t:tau}^gamma = sum_{d=0}^{delta-1} gamma^d r_{t+d}
  # Compute via masking and discounting
  # For efficiency, compute for each (b,t) pair
  def compute_inter_reward(carry, inp):
    # Reverse scan: accumulate from tau back to t
    agg = carry  # [B]
    rew_t, disc_t = inp  # [B], [B]
    agg = rew_t + disc_t * agg
    return agg, agg

  # We need per-timestep accumulated rewards.
  # Use the gate changes to reset accumulation (like THICK TF).
  change_disc = (1.0 - f32(changes)) * discount  # [B, T] - discount except at changes
  _, inter_rewards = jax.lax.scan(
      compute_inter_reward,
      jnp.zeros(B),
      (jnp.moveaxis(rewards, 1, 0), jnp.moveaxis(change_disc, 1, 0)),
      reverse=True)
  inter_rewards = jnp.moveaxis(inter_rewards, 0, 1)  # [B, T]

  targets = dict(
      stoch=target_stoch,
      action=target_action,
      context=target_context,
      time_delta=time_delta,
      inter_reward=inter_rewards,
  )
  return targets, valid


class HLWM(nj.Module):
  """High-Level World Model (Eq. 12-18).

  Learns to predict transitions between context change points using
  a posterior/prior over high-level actions.
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
    """Posterior Q_theta (Eq. 12): [c_t, z_t, c_tau, z_tau] -> A_t logits."""
    stoch_t_flat = stoch_t.reshape((*stoch_t.shape[:-2], -1))
    stoch_tau_flat = stoch_tau.reshape((*stoch_tau.shape[:-2], -1))
    x = jnp.concatenate([context_t, stoch_t_flat, context_tau, stoch_tau_flat], -1)
    x = self._body('post', x)
    logit = self.sub('post_logit', nn.Linear, self.hl_act_dim, **self.kw)(x)
    return logit

  def _prior(self, context_t, stoch_t):
    """Prior P_theta (Eq. 15): [c_t, z_t] -> A_hat_t logits."""
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
    """Compute HLWM losses (Eq. 18).

    Args:
      feat: dict with context, stoch, gates from C-RSSM observe
      actions: dict of action tensors (will be embedded via DictConcat)
      rewards: [B,T] rewards
      discount: scalar
      training: bool

    Returns:
      losses: dict of [B,T] loss tensors
      metrics: dict of scalar metrics
    """
    losses = {}
    metrics = {}

    # Embed dict actions into a flat tensor for indexing in generate_targets
    act_emb = nn.DictConcat(self.act_space, 1)(actions)
    targets, valid = generate_targets(feat, act_emb, rewards, discount)
    valid_f = f32(valid)  # [B, T]

    B, T = valid.shape
    context_t = sg(feat['context'])
    stoch_t = sg(feat['stoch'])

    # Posterior and prior HL actions
    post_logit = self._posterior(
        context_t, stoch_t, sg(targets['context']), sg(targets['stoch']))
    prior_logit = self._prior(context_t, stoch_t)

    # Sample from posterior for prediction heads
    post_dist = embodied.jax.outs.OneHot(post_logit, 0.01)
    prior_dist = embodied.jax.outs.OneHot(prior_logit, 0.01)
    hl_act = nn.cast(post_dist.sample(seed=nj.seed()))

    # Prediction heads from [A_t, c_t, z_t]
    pred_feat = self._predict('pred', hl_act, context_t, stoch_t)

    # State prediction (Eq. 14): predict target stoch logits
    stoch_logit = self.sub(
        'stoch_out', nn.Linear, self.stoch * self.classes, **self.kw)(pred_feat)
    stoch_logit = stoch_logit.reshape((*stoch_logit.shape[:-1], self.stoch, self.classes))
    target_stoch_logit = sg(targets['stoch'])  # [B, T, S, C] one-hot
    stoch_dist = embodied.jax.outs.OneHot(stoch_logit, 0.01)
    stoch_dist = embodied.jax.outs.Agg(stoch_dist, 1, jnp.sum)
    target_dist = embodied.jax.outs.OneHot(target_stoch_logit, 0.01)
    target_dist = embodied.jax.outs.Agg(target_dist, 1, jnp.sum)
    losses['hlwm_stoch'] = target_dist.kl(stoch_dist) * valid_f

    # Action prediction (Eq. 13)
    act_pred = self.sub('act_out', nn.Linear, act_emb.shape[-1], **self.kw)(pred_feat)
    losses['hlwm_action'] = jnp.square(act_pred - sg(targets['action'])).sum(-1) * valid_f

    # Time prediction (Eq. 16)
    time_pred = self.sub('time_out', nn.Linear, 1, **self.kw)(pred_feat)
    time_pred = time_pred.squeeze(-1)
    losses['hlwm_time'] = jnp.square(
        time_pred - sg(targets['time_delta'])) * valid_f

    # Reward prediction (Eq. 17)
    rew_pred = self.sub('rew_out', nn.Linear, 1, **self.kw)(pred_feat)
    rew_pred = rew_pred.squeeze(-1)
    losses['hlwm_reward'] = jnp.square(
        rew_pred - sg(targets['inter_reward'])) * valid_f

    # HL action KL (Eq. 18): KL(Q || P)
    act_kl = post_dist.kl(prior_dist)
    losses['hlwm_act_kl'] = act_kl * valid_f

    metrics['hlwm_valid_frac'] = valid_f.mean()
    metrics['hlwm_time_pred'] = time_pred.mean()
    metrics['hlwm_prior_ent'] = prior_dist.entropy().mean()
    metrics['hlwm_post_ent'] = post_dist.entropy().mean()

    return losses, metrics

  def predict(self, coarse_feat, stoch, context, training):
    """Sample from prior and predict outcomes for V^long.

    Args:
      coarse_feat: [B, D] coarse features (not used directly, kept for API)
      stoch: [B, S, C] current stochastic state
      context: [B, m] current context

    Returns:
      dict with predicted reward, time_delta, stoch_logit
    """
    prior_logit = self._prior(context, stoch)
    prior_dist = embodied.jax.outs.OneHot(prior_logit, 0.01)
    hl_act = nn.cast(prior_dist.sample(seed=nj.seed()))

    pred_feat = self._predict('pred', hl_act, context, stoch)

    # Predictions
    stoch_logit = self.sub(
        'stoch_out', nn.Linear, self.stoch * self.classes, **self.kw)(pred_feat)
    stoch_logit = stoch_logit.reshape(
        (*stoch_logit.shape[:-1], self.stoch, self.classes))

    time_pred = self.sub('time_out', nn.Linear, 1, **self.kw)(pred_feat)
    time_pred = jax.nn.relu(time_pred.squeeze(-1)) + 1  # at least 1 step

    rew_pred = self.sub('rew_out', nn.Linear, 1, **self.kw)(pred_feat)
    rew_pred = rew_pred.squeeze(-1)

    return dict(
        stoch_logit=stoch_logit,
        time_delta=time_pred,
        reward=rew_pred,
    )
