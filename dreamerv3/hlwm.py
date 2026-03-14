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
  - z_{tau-1}: stoch logits at timestep before the boundary
  - a_{tau-1}: action at timestep before the boundary
  - c_tau, z_tau: context and stoch at the boundary (for posterior)
  - Time delta = tau(t) - t
  - Accumulated discounted reward from t to tau(t)

  Args:
    feat: dict with 'context' [B,T,m], 'stoch' [B,T,S,C],
          'logit' [B,T,S,C], 'gate_prob' [B,T]
    actions: [B,T,A] embedded actions
    rewards: [B,T] rewards
    discount: scalar discount factor

  Returns:
    targets: dict of target tensors
    valid: [B,T] boolean mask (False where no future change in sequence)
  """
  context = feat['context']  # [B, T, m]
  stoch = feat['stoch']      # [B, T, S, C]
  logit = feat['logit']      # [B, T, S, C]
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

  b_idx = jnp.arange(B)[:, None].repeat(T, 1)  # [B, T]

  # tau_m1 = max(tau_abs - 1, 0): timestep before boundary
  tau_m1 = jnp.maximum(tau_abs - 1, 0)

  # Gather z_{tau-1} logits and a_{tau-1}
  target_stoch_logit = logit[b_idx, tau_m1]  # [B, T, S, C]
  target_action = actions[b_idx, tau_m1]     # [B, T, A]

  # Gather c_tau and z_tau (for posterior)
  target_context_tau = context[b_idx, tau_abs]  # [B, T, m]
  target_stoch_tau = stoch[b_idx, tau_abs]      # [B, T, S, C]

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
      stoch_logit=target_stoch_logit,
      action=target_action,
      context_tau=target_context_tau,
      stoch_tau=target_stoch_tau,
      time_delta=time_delta,
      inter_reward=inter_rewards,
  )
  return targets, valid


class HLWM(nj.Module):
  """High-Level World Model.

  Predicts z_{tau-1} and a_{tau-1} at the next boundary, then runs
  the coarse GRU to produce c_tau for the coarse prior.
  """

  hl_act_dim: int = 5
  layers: int = 3
  units: int = 256
  act: str = 'silu'
  norm: str = 'rms'

  def __init__(self, stoch, classes, context, act_space, action_dim, **kw):
    self.stoch = stoch
    self.classes = classes
    self.context_dim = context
    self.stoch_dim = stoch * classes
    self.action_dim = action_dim
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

  def _stoch_dist(self, logit):
    """OneHot distribution over stoch logits, matching CRSSM."""
    out = embodied.jax.outs.OneHot(logit, 0.01)
    out = embodied.jax.outs.Agg(out, 1, jnp.sum)
    return out

  def loss(self, feat, actions, rewards, discount, training):
    """Compute HLWM losses.

    Args:
      feat: dict with context, stoch, logit, gate_prob from C-RSSM observe
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

    # Posterior and prior HL actions
    post_logit = self._posterior(
        context_t, stoch_t,
        sg(targets['context_tau']), sg(targets['stoch_tau']))
    prior_logit = self._prior(context_t, stoch_t)

    post_dist = embodied.jax.outs.OneHot(post_logit, 0.01)
    prior_dist = embodied.jax.outs.OneHot(prior_logit, 0.01)
    hl_act = nn.cast(post_dist.sample(seed=nj.seed()))

    # Prediction heads from [A_t, c_t, z_t]
    pred_feat = self._predict('pred', hl_act, context_t, stoch_t)

    # Stoch prediction: KL(sg(target_dist) || pred_dist)
    stoch_logit_pred = self.sub(
        'stoch_out', nn.Linear, self.stoch_dim, **self.kw)(pred_feat)
    stoch_logit_pred = stoch_logit_pred.reshape(
        (*stoch_logit_pred.shape[:-1], self.stoch, self.classes))
    pred_stoch_dist = self._stoch_dist(stoch_logit_pred)
    target_stoch_dist = self._stoch_dist(sg(targets['stoch_logit']))
    losses['hlwm_stoch'] = target_stoch_dist.kl(pred_stoch_dist) * valid_f

    # Action prediction: -log_prob(target)
    action_pred = self.sub(
        'action_out', nn.Linear, self.action_dim, **self.kw)(pred_feat)
    losses['hlwm_action'] = -jax.nn.log_softmax(action_pred) * sg(targets['action'])
    # Sum over action dim to get scalar per timestep
    losses['hlwm_action'] = losses['hlwm_action'].sum(-1) * valid_f

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
    """Sample from prior and predict z_{tau-1}, a_{tau-1}, reward, time_delta.

    Args:
      context: [B, m] current context
      stoch: [B, S, C] current stochastic state

    Returns:
      dict with sampled stoch, action, reward, time_delta
    """
    prior_logit = self._prior(context, stoch)
    prior_dist = embodied.jax.outs.OneHot(prior_logit, 0.01)
    hl_act = nn.cast(prior_dist.sample(seed=nj.seed()))

    pred_feat = self._predict('pred', hl_act, context, stoch)

    # Stoch prediction: sample z_{tau-1}
    stoch_logit_pred = self.sub(
        'stoch_out', nn.Linear, self.stoch_dim, **self.kw)(pred_feat)
    stoch_logit_pred = stoch_logit_pred.reshape(
        (*stoch_logit_pred.shape[:-1], self.stoch, self.classes))
    pred_stoch_dist = self._stoch_dist(stoch_logit_pred)
    pred_stoch = nn.cast(pred_stoch_dist.sample(seed=nj.seed()))
    pred_stoch_flat = pred_stoch.reshape((*pred_stoch.shape[:-2], -1))

    # Action prediction: sample a_{tau-1}
    action_logit = self.sub(
        'action_out', nn.Linear, self.action_dim, **self.kw)(pred_feat)
    action_dist = embodied.jax.outs.OneHot(action_logit, 0.01)
    pred_action = nn.cast(action_dist.sample(seed=nj.seed()))

    time_pred = self.sub('time_out', nn.Linear, 1, **self.kw)(pred_feat)
    time_pred = jax.nn.relu(time_pred.squeeze(-1)) + 1  # at least 1 step

    rew_pred = self.sub('rew_out', nn.Linear, 1, **self.kw)(pred_feat)
    rew_pred = rew_pred.squeeze(-1)

    return dict(
        stoch=pred_stoch_flat,
        stoch_logit=stoch_logit_pred,
        action=pred_action,
        time_delta=time_pred,
        reward=rew_pred,
    )

  def predict_given_action(self, hl_act, context, stoch):
    """Predict outcomes for an explicit HL action (for tree search).

    Args:
      hl_act: [B, hl_act_dim] one-hot HL action
      context: [B, m] current context
      stoch: [B, S, C] current stochastic state

    Returns:
      dict with stoch (flat sampled), stoch_logit [B, S, C],
      action (sampled), time_delta, reward
    """
    pred_feat = self._predict('pred', hl_act, context, stoch)

    # Stoch prediction
    stoch_logit_pred = self.sub(
        'stoch_out', nn.Linear, self.stoch_dim, **self.kw)(pred_feat)
    stoch_logit_pred = stoch_logit_pred.reshape(
        (*stoch_logit_pred.shape[:-1], self.stoch, self.classes))
    pred_stoch_dist = self._stoch_dist(stoch_logit_pred)
    pred_stoch = nn.cast(pred_stoch_dist.sample(seed=nj.seed()))
    pred_stoch_flat = pred_stoch.reshape((*pred_stoch.shape[:-2], -1))

    # Action prediction
    action_logit = self.sub(
        'action_out', nn.Linear, self.action_dim, **self.kw)(pred_feat)
    action_dist = embodied.jax.outs.OneHot(action_logit, 0.01)
    pred_action = nn.cast(action_dist.sample(seed=nj.seed()))

    time_pred = self.sub('time_out', nn.Linear, 1, **self.kw)(pred_feat)
    time_pred = jax.nn.relu(time_pred.squeeze(-1)) + 1

    rew_pred = self.sub('rew_out', nn.Linear, 1, **self.kw)(pred_feat)
    rew_pred = rew_pred.squeeze(-1)

    return dict(
        stoch=pred_stoch_flat,
        stoch_logit=stoch_logit_pred,
        action=pred_action,
        time_delta=time_pred,
        reward=rew_pred,
    )
