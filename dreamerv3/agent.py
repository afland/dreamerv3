import re

import chex
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

from . import crssm
from . import hlwm as hlwm_mod
from . import rssm

f32 = jnp.float32
i32 = jnp.int32
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)
prefix = lambda xs, p: {f'{p}/{k}': v for k, v in xs.items()}
concat = lambda xs, a: jax.tree.map(lambda *x: jnp.concatenate(x, a), *xs)
isimage = lambda s: s.dtype == np.uint8 and len(s.shape) == 3


class Agent(embodied.jax.Agent):

  banner = [
      r"---  ___                           __   ______ ---",
      r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---",
      r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---",
      r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---",
  ]

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space
    self.config = config

    exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
    enc_space = {k: v for k, v in obs_space.items() if k not in exclude}
    dec_space = {k: v for k, v in obs_space.items() if k not in exclude}
    self.enc = {
        'simple': rssm.Encoder,
    }[config.enc.typ](enc_space, **config.enc[config.enc.typ], name='enc')
    dyn_kw = dict(config.dyn[config.dyn.typ])
    self.dyn = {
        'rssm': rssm.RSSM,
        'crssm': crssm.CRSSM,
    }[config.dyn.typ](act_space, **dyn_kw, name='dyn')
    self.dec = {
        'simple': rssm.Decoder,
    }[config.dec.typ](dec_space, **config.dec[config.dec.typ], name='dec')

    if config.dyn.typ == 'crssm':
      if config.context_in_policy:
        # Actor/critic see [deter, stoch, context]
        self.feat2tensor = lambda x: jnp.concatenate([
            nn.cast(x['deter']),
            nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1))),
            nn.cast(x['context'])], -1)
      else:
        # Fine pathway only: [deter, stoch]
        self.feat2tensor = lambda x: jnp.concatenate([
            nn.cast(x['deter']),
            nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1)))], -1)
      # Coarse pathway: context + time_delta
      self.coarse_feat2tensor = lambda x: jnp.concatenate([
          nn.cast(x['context']),
          nn.cast(x['time_delta'])[..., None]], -1)
    else:
      self.feat2tensor = lambda x: jnp.concatenate([
          nn.cast(x['deter']),
          nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1)))], -1)
      self.coarse_feat2tensor = None

    if config.thick.enabled:
      _base = self.feat2tensor
      if config.thick.goal_type == 'c':
        self.pol_feat2tensor = lambda x: jnp.concatenate([
            _base(x), nn.cast(x['goal'])], -1)
      else:
        self.pol_feat2tensor = lambda x: jnp.concatenate([
            _base(x),
            nn.cast(x['goal'].reshape((*x['goal'].shape[:-2], -1)))], -1)
    else:
      self.pol_feat2tensor = self.feat2tensor

    scalar = elements.Space(np.float32, ())
    binary = elements.Space(bool, (), 0, 2)
    self.rew = embodied.jax.MLPHead(scalar, **config.rewhead, name='rew')
    self.con = embodied.jax.MLPHead(binary, **config.conhead, name='con')

    # Coarse prediction heads (only for C-RSSM)
    if config.dyn.typ == 'crssm':
      self.coarse_rew = embodied.jax.MLPHead(
          scalar, **config.coarse_rewhead, name='coarse_rew')
      self.coarse_con = embodied.jax.MLPHead(
          binary, **config.coarse_conhead, name='coarse_con')
      self.coarse_dec = rssm.CoarseDecoder(
          dec_space, **config.coarse_dec, name='coarse_dec')
    else:
      self.coarse_rew = None
      self.coarse_con = None
      self.coarse_dec = None

    d1, d2 = config.policy_dist_disc, config.policy_dist_cont
    outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
    self.pol = embodied.jax.MLPHead(
        act_space, outs, **config.policy, name='pol')

    self.val = embodied.jax.MLPHead(scalar, **config.value, name='val')
    self.slowval = embodied.jax.SlowModel(
        embodied.jax.MLPHead(scalar, **config.value, name='slowval'),
        source=self.val, **config.slowvalue)

    self.retnorm = embodied.jax.Normalize(**config.retnorm, name='retnorm')
    self.valnorm = embodied.jax.Normalize(**config.valnorm, name='valnorm')
    self.advnorm = embodied.jax.Normalize(**config.advnorm, name='advnorm')

    # THICK components
    if config.thick.enabled:
      assert config.dyn.typ == 'crssm', 'THICK requires C-RSSM'
      dyn_cfg = config.dyn[config.dyn.typ]
      # Compute action embedding dim from act_space
      action_dim = sum(
          np.asarray(v.classes).flatten()[0].item() if v.discrete
          else int(np.prod(v.shape))
          for v in act_space.values())
      self.hlwm = hlwm_mod.HLWM(
          stoch=dyn_cfg.stoch, classes=dyn_cfg.classes,
          context=dyn_cfg.context, act_space=act_space,
          action_dim=action_dim,
          hl_act_cats=config.thick.hl_act_cats,
          hl_act_classes=config.thick.hl_act_classes,
          segment_length=dyn_cfg.get('segment_length', 0),
          use_logits=config.thick.hlwm_use_logits,
          mgr_policy=dict(config.thick.mgr_policy),
          **config.thick.hlwm, name='hlwm')
      self.coarse_val = embodied.jax.MLPHead(
          scalar, **config.value, name='coarse_val')
      self.slow_coarse_val = embodied.jax.SlowModel(
          embodied.jax.MLPHead(scalar, **config.value, name='slow_coarse_val'),
          source=self.coarse_val, **config.slowvalue)
      self.mgr_retnorm = embodied.jax.Normalize(**config.retnorm, name='mgr_retnorm')
      self.mgr_valnorm = embodied.jax.Normalize(**config.valnorm, name='mgr_valnorm')
      self.mgr_advnorm = embodied.jax.Normalize(**config.advnorm, name='mgr_advnorm')
    else:
      self.hlwm = None
      self.coarse_val = None
      self.slow_coarse_val = None
      self.mgr_retnorm = None
      self.mgr_valnorm = None
      self.mgr_advnorm = None

    self.modules = [
        self.dyn, self.enc, self.dec, self.rew, self.con, self.pol, self.val]
    if self.coarse_rew:
      self.modules.extend([self.coarse_rew, self.coarse_con])
      if config.loss_scales.get('coarse_rec', 0.0) != 0.0:
        self.modules.append(self.coarse_dec)
    if self.hlwm:
      self.modules.append(self.hlwm)
    if self.coarse_val:
      self.modules.append(self.coarse_val)

    self.opt = embodied.jax.Optimizer(
        self.modules, self._make_opt(**config.opt), summary_depth=1,
        name='opt')

    scales = self.config.loss_scales.copy()
    rec = scales.pop('rec')
    scales.update({k: rec for k in dec_space})
    # Remove scales not needed for current config
    if config.dyn.typ != 'crssm':
      for k in ('coarse_dyn', 'sparse', 'gate_info',
                'coarse_rec', 'coarse_rew', 'coarse_con', 'gate_improve',
                'refractory'):
        scales.pop(k, None)
    if config.dyn.typ == 'crssm' and config.dyn.crssm.get('segment_length', 0) > 0:
      for k in ('sparse', 'gate_info', 'gate_improve', 'refractory', 'hlwm_time'):
        scales.pop(k, None)
    if not config.thick.enabled:
      for k in ('hlwm_stoch', 'hlwm_action', 'hlwm_time',
                'hlwm_reward', 'hlwm_act_kl', 'coarse_val', 'mgr_policy'):
        scales.pop(k, None)
    self.scales = {k: v for k, v in scales.items() if float(v) != 0.0}

  @staticmethod
  def _cosine_sim(a, b):
    # Max-cosine: normalize both by max(||a||, ||b||) so magnitude mismatch is penalized.
    a = a.reshape((*a.shape[:2], -1))
    b = b.reshape((*b.shape[:2], -1))
    norm = jnp.maximum(jnp.linalg.norm(a, axis=-1, keepdims=True),
                       jnp.linalg.norm(b, axis=-1, keepdims=True)) + 1e-8
    return (a / norm * b / norm).sum(-1)

  def _imagine_with_goals(self, starts, initial_goal, H, training):
    """Custom imagination loop that threads goals through policy input.

    Manager picks new goal at gate fires (stop-gradient: don't train manager
    through worker).
    Args:
      initial_goal: [BK, S, C] (z goals) or [BK, m] (c goals)
    """
    BK = starts['deter'].shape[0]

    def step(carry, _):
      crssm_carry, current_goal = carry
      feat_for_pol = {**sg(crssm_carry), 'goal': current_goal}
      action = sample(self.pol(self.pol_feat2tensor(feat_for_pol), 1))
      crssm_carry, (feat, act) = self.dyn.imagine(
          crssm_carry, action, 1, training, single=True)

      # On gate fire: manager picks new goal (stop-gradient)
      gate_bin = feat['gate_binary']
      ctx = crssm_carry['context']
      z = crssm_carry['logit'] if self.config.thick.hlwm_use_logits else crssm_carry['stoch']
      mgr_logits = sg(self.hlwm._manager(ctx, z))
      hl_act = sg(self.hlwm._hl_act_sample(mgr_logits))
      preds = sg(self.hlwm.predict_given_action(hl_act, ctx, z))

      if self.config.thick.goal_type == 'z':
        new_goal = preds['stoch_logit']
        current_goal = jnp.where(gate_bin[:, None, None] > 0.5, new_goal, current_goal)
      else:
        new_ctx = sg(self.dyn.context_step(ctx, preds['stoch'], preds['action']))
        current_goal = jnp.where(gate_bin[:, None] > 0.5, new_ctx, current_goal)

      feat = {**feat, 'goal': current_goal}
      return (crssm_carry, current_goal), (feat, act)

    init = (nn.cast(starts), initial_goal)
    final, (imgfeat, imgact) = nj.scan(step, init, (), H, unroll=1, axis=1)
    return final[0], imgfeat, imgact

  def _manager_imagine(self, starts, training):
    """Coarse imagination for manager actor-critic.

    Returns:
        mgr_ctx: [BK, M+1, m] contexts
        mgr_z: [BK, M+1, S, C] stochastic states
        mgr_act: [BK, M+1, cats*classes] manager actions (flat one-hot)
        mgr_rew: [BK, M+1] HLWM rewards (zero at t=0)
        mgr_logits: [BK, M+1, cats, classes] logits for policy gradient
    """
    M = self.config.thick.mgr_imag_length
    BK = starts['context'].shape[0]

    c0 = starts['context']
    z0 = starts['logit'] if self.config.thick.hlwm_use_logits else starts['stoch']

    contexts = [c0]
    stochs = [z0]
    actions = []
    rewards = []
    logits_list = []

    c, z = c0, z0
    for k in range(M):
      mgr_logits = self.hlwm._manager(c, z)
      hl_act = self.hlwm._hl_act_sample(mgr_logits)

      preds = self.hlwm.predict_given_action(hl_act, c, z)
      pred_stoch_flat = nn.cast(preds['stoch'])
      pred_action = nn.cast(preds['action'])

      c_new = self.dyn.context_step(c, pred_stoch_flat, pred_action)
      td_zero = nn.cast(jnp.zeros(BK, f32))
      z_logit = self.dyn._coarse_prior(c_new, sg(pred_stoch_flat), sg(pred_action), td_zero)
      z_new = nn.cast(self.dyn._dist(z_logit).sample(seed=nj.seed()))

      actions.append(hl_act)
      rewards.append(preds['reward'])
      logits_list.append(mgr_logits)
      contexts.append(c_new)
      stochs.append(z_new)

      c, z = c_new, z_new

    mgr_ctx = jnp.stack(contexts, 1)
    mgr_z = jnp.stack(stochs, 1)
    mgr_act = jnp.stack(actions, 1)
    mgr_rew = jnp.stack(rewards, 1)
    mgr_logits = jnp.stack(logits_list, 1)

    # Pad with dummy first timestep
    mgr_act = jnp.concatenate([jnp.zeros((BK, 1, mgr_act.shape[-1])), mgr_act], 1)
    mgr_rew = jnp.concatenate([jnp.zeros((BK, 1)), mgr_rew], 1)
    mgr_logits = jnp.concatenate([jnp.zeros((BK, 1, *mgr_logits.shape[2:])), mgr_logits], 1)

    return mgr_ctx, mgr_z, mgr_act, mgr_rew, mgr_logits

  @property
  def policy_keys(self):
    if self.config.thick.enabled:
      return '^(enc|dyn|dec|pol|hlwm|coarse_val)/'
    return '^(enc|dyn|dec|pol)/'

  def _coarse_critic_inp(self, feat):
    """Build [context, stoch] input for coarse critic."""
    return jnp.concatenate([
        nn.cast(feat['context']),
        nn.cast(feat['stoch'].reshape((*feat['stoch'].shape[:-2], -1)))], -1)

  @property
  def ext_space(self):
    spaces = {}
    spaces['consec'] = elements.Space(np.int32)
    spaces['stepid'] = elements.Space(np.uint8, 20)
    if self.config.replay_context:
      spaces.update(elements.tree.flatdict(dict(
          enc=self.enc.entry_space,
          dyn=self.dyn.entry_space,
          dec=self.dec.entry_space)))
    return spaces

  def init_policy(self, batch_size):
    zeros = lambda x: jnp.zeros((batch_size, *x.shape), x.dtype)
    carry = (
        self.enc.initial(batch_size),
        self.dyn.initial(batch_size),
        self.dec.initial(batch_size),
        jax.tree.map(zeros, self.act_space))
    if self.config.thick.enabled:
      if self.config.thick.goal_type == 'c':
        m = self.config.dyn[self.config.dyn.typ].context
        carry = carry + (jnp.zeros((batch_size, m), f32),)
      else:
        S = self.config.dyn[self.config.dyn.typ].stoch
        C = self.config.dyn[self.config.dyn.typ].classes
        carry = carry + (jnp.zeros((batch_size, S, C), f32),)
    return carry

  def init_train(self, batch_size):
    return self.init_policy(batch_size)

  def init_report(self, batch_size):
    return self.init_policy(batch_size)

  def policy(self, carry, obs, mode='train'):
    if self.config.thick.enabled:
      (enc_carry, dyn_carry, dec_carry, prevact, goal) = carry
    else:
      (enc_carry, dyn_carry, dec_carry, prevact) = carry
    kw = dict(training=False, single=True)
    reset = obs['is_first']
    enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, **kw)
    dyn_carry, dyn_entry, feat = self.dyn.observe(
        dyn_carry, tokens, prevact, reset, **kw)
    dec_entry = {}
    if dec_carry:
      dec_carry, dec_entry, recons = self.dec(dec_carry, feat, reset, **kw)
    if self.hlwm:
      # Replan on gate fire: manager picks new goal
      gate_fired = feat['gate_binary'] > 0.5  # [B]
      stoch_inp = feat['logit'] if self.config.thick.hlwm_use_logits else feat['stoch']
      mgr_logits = self.hlwm._manager(feat['context'], stoch_inp)
      hl_act = self.hlwm._hl_act_sample(mgr_logits)
      preds = self.hlwm.predict_given_action(hl_act, feat['context'], stoch_inp)
      if self.config.thick.goal_type == 'c':
        new_ctx = self.dyn.context_step(feat['context'], preds['stoch'], preds['action'])
        new_goal = new_ctx  # [B, m]
        goal = jnp.where(gate_fired[:, None], new_goal, goal)
      else:
        new_goal = preds['stoch_logit']  # [B, S, C]
        goal = jnp.where(gate_fired[:, None, None], new_goal, goal)
      feat = {**feat, 'goal': goal}
    policy = self.pol(self.pol_feat2tensor(feat), bdims=1)
    act = sample(policy)
    out = {}
    out['finite'] = elements.tree.flatdict(jax.tree.map(
        lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
        dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act)))
    if 'context' in feat:
      out['context'] = feat['context']
    if 'gate_prob' in feat:
      out['gate_prob'] = feat['gate_prob']
    carry = (enc_carry, dyn_carry, dec_carry, act)
    if self.config.thick.enabled:
      carry = carry + (goal,)
    if self.config.replay_context:
      out.update(elements.tree.flatdict(dict(
          enc=enc_entry, dyn=dyn_entry, dec=dec_entry)))
    return carry, act, out

  def train(self, carry, data):
    carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
    metrics, (carry, entries, outs, mets) = self.opt(
        self.loss, carry, obs, prevact, training=True, has_aux=True)
    metrics.update(mets)
    self.slowval.update()
    if self.slow_coarse_val:
      self.slow_coarse_val.update()
    outs = {}
    if self.config.replay_context:
      updates = elements.tree.flatdict(dict(
          stepid=stepid, enc=entries[0], dyn=entries[1], dec=entries[2]))
      B, T = obs['is_first'].shape
      assert all(x.shape[:2] == (B, T) for x in updates.values()), (
          (B, T), {k: v.shape for k, v in updates.items()})
      outs['replay'] = updates
    # if self.config.replay.fracs.priority > 0:
    #   outs['replay']['priority'] = losses['model']
    carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
    if self.config.thick.enabled:
      B = data['is_first'].shape[0]
      if self.config.thick.goal_type == 'c':
        m = self.config.dyn[self.config.dyn.typ].context
        carry = carry + (jnp.zeros((B, m), f32),)
      else:
        S = self.config.dyn[self.config.dyn.typ].stoch
        C = self.config.dyn[self.config.dyn.typ].classes
        carry = carry + (jnp.zeros((B, S, C), f32),)
    return carry, outs, metrics

  def loss(self, carry, obs, prevact, training):
    enc_carry, dyn_carry, dec_carry = carry
    reset = obs['is_first']
    B, T = reset.shape
    losses = {}
    metrics = {}

    # World model
    enc_carry, enc_entries, tokens = self.enc(
        enc_carry, obs, reset, training)
    dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
        dyn_carry, tokens, prevact, reset, training)
    # Pop extra tensors before any tree ops on repfeat
    ctx_before_gate = repfeat.pop('ctx_before_gate', None)
    ctx_after_gru = repfeat.pop('ctx_after_gru', None)
    time_delta_pre = repfeat.pop('time_delta_pre', None)
    surprise = repfeat.pop('surprise', None)
    # Drop losses not in scales (e.g. gate_info when scale=0.0)
    losses.update({k: v for k, v in los.items() if k in self.scales})
    metrics.update(mets)
    dec_carry, dec_entries, recons = self.dec(
        dec_carry, repfeat, reset, training)
    inp = self.feat2tensor(repfeat)
    losses['rew'] = self.rew(inp, 2).loss(obs['reward'])
    con = f32(~obs['is_terminal'])
    if self.config.contdisc:
      con *= 1 - 1 / self.config.horizon
    losses['con'] = self.con(self.feat2tensor(repfeat), 2).loss(con)
    for key, recon in recons.items():
      space, value = self.obs_space[key], obs[key]
      assert value.dtype == space.dtype, (key, space, value.dtype)
      target = f32(value) / 255 if isimage(space) else value
      losses[key] = recon.loss(sg(target))

    # Coarse prediction heads (C-RSSM only)
    # Full gradients flow to context/stoch (matching thix/THICK)
    if self.coarse_rew:
      use_rew = 'coarse_rew' in self.scales
      use_con = 'coarse_con' in self.scales
      use_rec = 'coarse_rec' in self.scales
      if use_rew or use_con or use_rec:
        coarse_inp = self.coarse_feat2tensor(repfeat)
        if use_rew:
          losses['coarse_rew'] = self.coarse_rew(coarse_inp, 2).loss(obs['reward'])
        if use_con:
          losses['coarse_con'] = self.coarse_con(coarse_inp, 2).loss(con)
        if use_rec:
          coarse_dec_losses = self.coarse_dec(coarse_inp, 2, obs)
          losses['coarse_rec'] = sum(coarse_dec_losses.values())

    # Gate improvement: reward gate for firing where context update helps
    # Always compute when crssm (for logging), only add loss when scale present
    if ctx_before_gate is not None:
      gate_prob = repfeat['gate_prob']
      ctx_old = sg(ctx_before_gate)
      ctx_new = sg(ctx_after_gru)
      post = repfeat['logit']
      td_stale = nn.cast(time_delta_pre)  # pre-reset: actual staleness of old ctx
      td_zero = nn.cast(jnp.zeros_like(gate_prob))
      # Main coarse prior improvement (always available)
      z_flat = sg(repfeat['stoch'].reshape((*repfeat['stoch'].shape[:-2], -1)))
      actemb = nn.DictConcat(self.act_space, 1)(prevact)
      actemb /= sg(jnp.maximum(1, jnp.abs(actemb)))
      main_logit_old = self.dyn._coarse_prior(ctx_old, z_flat, actemb, td_stale)
      main_logit_new = self.dyn._coarse_prior(ctx_new, z_flat, actemb, td_zero)
      improve_main = (self.dyn._dist(sg(post)).kl(self.dyn._dist(main_logit_old))
                      - self.dyn._dist(sg(post)).kl(self.dyn._dist(main_logit_new)))
      improvement = improve_main
      metrics['improve_main'] = improve_main.mean()
      if 'gate_improve' in self.scales:
        losses['gate_improve'] = -gate_prob * sg(jax.nn.relu(improvement))
      metrics['gate_improve_mean'] = improvement.mean()
      metrics['gate_improve_std'] = improvement.std(-1).mean()
      metrics['gate_improve_pos_frac'] = f32(improvement > 0).mean()

    # HLWM losses (THICK only, stop-gradient inputs to match paper)
    if self.hlwm:
      hlwm_losses, hlwm_mets = self.hlwm.loss(
          sg(repfeat), sg(prevact), sg(obs['reward']),
          1 - 1 / self.config.horizon, training)
      # Gate HLWM losses by hlwm_start
      hlwm_mask = f32(self.opt.step.read() >= self.config.thick.hlwm_start)
      for k in hlwm_losses:
        hlwm_losses[k] = hlwm_losses[k] * hlwm_mask
      losses.update(hlwm_losses)
      metrics.update(prefix(hlwm_mets, 'hlwm'))

    B, T = reset.shape
    shapes = {k: v.shape for k, v in losses.items()}
    assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

    # Imagination
    K = min(self.config.imag_last or T, T)
    H = self.config.imag_length
    starts = self.dyn.starts(dyn_entries, dyn_carry, K)

    plan_mets = {}
    if self.hlwm:
      # Phase A: Manager coarse imagination + actor-critic
      hlwm_mask = f32(self.opt.step.read() >= self.config.thick.hlwm_start)

      mgr_ctx, mgr_z, mgr_act, mgr_rew, mgr_logits = \
          self._manager_imagine(starts, training)

      mgr_z_flat = mgr_z.reshape((*mgr_z.shape[:-2], -1))
      mgr_critic_inp = jnp.concatenate([mgr_ctx, mgr_z_flat], -1)

      mgr_los, mgr_mets = mgr_imag_loss(
          mgr_act, mgr_rew * hlwm_mask, mgr_logits,
          self.coarse_val(mgr_critic_inp, 2),
          self.slow_coarse_val(mgr_critic_inp, 2),
          self.mgr_retnorm, self.mgr_valnorm, self.mgr_advnorm,
          self.hlwm, update=training,
          horizon=self.config.horizon,
          actent=self.config.thick.mgr_actent,
          lam=self.config.imag_loss.lam,
          slowreg=self.config.imag_loss.slowreg)
      for k, v in mgr_los.items():
        losses[k] = v.mean(1).reshape((B, K))
      metrics.update(prefix(mgr_mets, 'mgr'))

      # Phase B: Get initial goal from manager for worker
      starts_ctx = starts['context']
      starts_z = starts['logit'] if self.config.thick.hlwm_use_logits else starts['stoch']
      init_mgr_logits = self.hlwm._manager(starts_ctx, starts_z)
      init_hl_act = sg(self.hlwm._hl_act_sample(init_mgr_logits))
      init_preds = self.hlwm.predict_given_action(init_hl_act, starts_ctx, starts_z)

      if self.config.thick.goal_type == 'z':
        initial_goal = sg(init_preds['stoch_logit'])  # [BK, S, C]
      else:
        c_goal = self.dyn.context_step(starts_ctx, init_preds['stoch'], init_preds['action'])
        initial_goal = sg(c_goal)  # [BK, m]

      # Phase C: Worker imagination with goals
      _, imgfeat, imgprevact = self._imagine_with_goals(
          starts, initial_goal, H, training)

      first = jax.tree.map(
          lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])), repfeat)
      if initial_goal.ndim == 2:
        first_goal = initial_goal[:, None]
      else:
        first_goal = initial_goal[:, None]
      first = {**sg(first, skip=self.config.ac_grads),
               'goal': first_goal}
      imgfeat = concat([first, sg(imgfeat)], 1)

      last_feat = jax.tree.map(lambda x: x[:, -1], imgfeat)
      lastact = sample(self.pol(self.pol_feat2tensor(last_feat), 1))
      lastact = jax.tree.map(lambda x: x[:, None], lastact)
      imgact = concat([imgprevact, lastact], 1)

      # Worker reward: dense cosine similarity
      if self.config.thick.goal_type == 'c':
        sim = self._cosine_sim(imgfeat['context'], sg(imgfeat['goal']))
      else:
        sim = self._cosine_sim(imgfeat['logit'], sg(imgfeat['goal']))

      inp = self.feat2tensor(imgfeat)
      rew = hlwm_mask * sim

      # Worker continuation: zero at gate fires
      worker_con = self.con(inp, 2).prob(1)
      if self.config.thick.worker_gate_cut:
        gate_bin = imgfeat['gate_binary']
        cut_mask = gate_bin.at[:, 0].set(0.0)
        worker_con = worker_con * (1 - cut_mask)

      plan_mets['plan/sim_mean'] = sim.mean()
      metrics.update(plan_mets)

      # Phase D: Worker imag_loss
      pol_inp = self.pol_feat2tensor(imgfeat)

      los, imgloss_out, mets = imag_loss(
          imgact, rew, worker_con,
          self.pol(pol_inp, 2),
          self.val(inp, 2),
          self.slowval(inp, 2),
          self.retnorm, self.valnorm, self.advnorm,
          update=training,
          contdisc=self.config.contdisc,
          horizon=self.config.horizon,
          **self.config.imag_loss)
      losses.update({k: v.mean(1).reshape((B, K)) for k, v in los.items()})
      metrics.update(mets)

    else:
      # Phase E: Non-THICK fallback
      policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))
      _, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training)
      first = jax.tree.map(
          lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])), repfeat)
      imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1)
      lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
      lastact = jax.tree.map(lambda x: x[:, None], lastact)
      imgact = concat([imgprevact, lastact], 1)
      inp = self.feat2tensor(imgfeat)
      rew = self.rew(inp, 2).pred()

      assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgfeat))
      assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgact))

      los, imgloss_out, mets = imag_loss(
          imgact, rew,
          self.con(inp, 2).prob(1),
          self.pol(self.feat2tensor(imgfeat), 2),
          self.val(inp, 2),
          self.slowval(inp, 2),
          self.retnorm, self.valnorm, self.advnorm,
          update=training,
          contdisc=self.config.contdisc,
          horizon=self.config.horizon,
          **self.config.imag_loss)
      losses.update({k: v.mean(1).reshape((B, K)) for k, v in los.items()})
      metrics.update(mets)

    # Replay
    if self.config.repval_loss:
      feat = sg(repfeat, skip=self.config.repval_grad)
      last, term, rew = [obs[k] for k in ('is_last', 'is_terminal', 'reward')]
      boot = imgloss_out['ret'][:, 0].reshape(B, K)
      feat, last, term, rew, boot = jax.tree.map(
          lambda x: x[:, -K:], (feat, last, term, rew, boot))
      inp = self.feat2tensor(feat)
      los, reploss_out, mets = repl_loss(
          last, term, rew, boot,
          self.val(inp, 2),
          self.slowval(inp, 2),
          self.valnorm,
          update=training,
          horizon=self.config.horizon,
          **self.config.repl_loss)
      losses.update(los)
      metrics.update(prefix(mets, 'reploss'))

    assert set(losses.keys()) == set(self.scales.keys()), (
        sorted(losses.keys()), sorted(self.scales.keys()))
    metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
    loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

    carry = (enc_carry, dyn_carry, dec_carry)
    entries = (enc_entries, dyn_entries, dec_entries)
    outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
    if surprise is not None:
      outs['surprise'] = surprise
    if ctx_before_gate is not None:
      outs['improvement'] = improvement
    return loss, (carry, entries, outs, metrics)

  def report(self, carry, data):
    if not self.config.report:
      return carry, {}

    carry, obs, prevact, _ = self._apply_replay_context(carry, data)
    (enc_carry, dyn_carry, dec_carry) = carry
    B, T = obs['is_first'].shape
    RB = min(6, B)
    metrics = {}

    # Train metrics
    _, (new_carry, entries, outs, mets) = self.loss(
        carry, obs, prevact, training=False)
    mets.update(mets)

    # Grad norms
    if self.config.report_gradnorms:
      for key in self.scales:
        try:
          lossfn = lambda data, carry: self.loss(
              carry, obs, prevact, training=False)[1][2]['losses'][key].mean()
          grad = nj.grad(lossfn, self.modules)(data, carry)[-1]
          metrics[f'gradnorm/{key}'] = optax.global_norm(grad)
        except KeyError:
          print(f'Skipping gradnorm summary for missing loss: {key}')

    # Open loop
    firsthalf = lambda xs: jax.tree.map(lambda x: x[:RB, :T // 2], xs)
    secondhalf = lambda xs: jax.tree.map(lambda x: x[:RB, T // 2:], xs)
    dyn_carry = jax.tree.map(lambda x: x[:RB], dyn_carry)
    dec_carry = jax.tree.map(lambda x: x[:RB], dec_carry)
    dyn_carry, _, obsfeat = self.dyn.observe(
        dyn_carry, firsthalf(outs['tokens']), firsthalf(prevact),
        firsthalf(obs['is_first']), training=False)
    _, imgfeat, _ = self.dyn.imagine(
        dyn_carry, secondhalf(prevact), length=T - T // 2, training=False)
    dec_carry, _, obsrecons = self.dec(
        dec_carry, obsfeat, firsthalf(obs['is_first']), training=False)
    dec_carry, _, imgrecons = self.dec(
        dec_carry, imgfeat, jnp.zeros_like(secondhalf(obs['is_first'])),
        training=False)

    # Video preds
    for key in self.dec.imgkeys:
      assert obs[key].dtype == jnp.uint8
      true = obs[key][:RB]
      pred = jnp.concatenate([obsrecons[key].pred(), imgrecons[key].pred()], 1)
      pred = jnp.clip(pred * 255, 0, 255).astype(jnp.uint8)
      error = ((i32(pred) - i32(true) + 255) / 2).astype(np.uint8)
      video = jnp.concatenate([true, pred, error], 2)

      video = jnp.pad(video, [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]])
      mask = jnp.zeros(video.shape, bool).at[:, :, 2:-2, 2:-2, :].set(True)
      border = jnp.full((T, 3), jnp.array([0, 255, 0]), jnp.uint8)
      border = border.at[T // 2:].set(jnp.array([255, 0, 0], jnp.uint8))
      video = jnp.where(mask, video, border[None, :, None, None, :])
      video = jnp.concatenate([video, 0 * video[:, :10]], 1)

      B, T, H, W, C = video.shape
      grid = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
      metrics[f'openloop/{key}'] = grid

    # Per-timestep gate probabilities (C-RSSM only)
    repfeat = outs.get('repfeat', {})
    if isinstance(repfeat, dict) and 'gate_prob' in repfeat:
      gp = repfeat['gate_prob']  # [B, T]
      gp_mean = gp.mean(0)  # [T]
      for t in range(gp_mean.shape[0]):
        metrics[f'report/boundprob_t{t:02d}'] = gp_mean[t]
    if 'surprise' in outs:
      s = outs['surprise'].mean(0)  # [T]
      for t in range(s.shape[0]):
        metrics[f'report/surprise_t{t:02d}'] = s[t]
    if 'improvement' in outs:
      imp = outs['improvement'].mean(0)  # [T]
      for t in range(imp.shape[0]):
        metrics[f'report/improvement_t{t:02d}'] = imp[t]
    # Per-timestep coarse losses
    report_losses = outs.get('losses', {})
    for lname in ('coarse_dyn', 'coarse_rew', 'coarse_con', 'coarse_rec'):
      if lname in report_losses:
        vals = report_losses[lname].mean(0)  # [T]
        for t in range(vals.shape[0]):
          metrics[f'report/{lname}_t{t:02d}'] = vals[t]

    # Per-episode gate probs and improvement (first 3 episodes)
    if isinstance(repfeat, dict) and 'gate_prob' in repfeat:
      gp = repfeat['gate_prob']  # [B, T]
      for ep in range(min(3, gp.shape[0])):
        for t in range(gp.shape[1]):
          metrics[f'report/gate_ep{ep}_t{t:02d}'] = gp[ep, t]
    if 'improvement' in outs:
      imp = outs['improvement']  # [B, T]
      for ep in range(min(3, imp.shape[0])):
        for t in range(imp.shape[1]):
          metrics[f'report/imp_ep{ep}_t{t:02d}'] = imp[ep, t]

    carry = (*new_carry, {k: data[k][:, -1] for k in self.act_space})
    if self.config.thick.enabled:
      B = data['is_first'].shape[0]
      if self.config.thick.goal_type == 'c':
        m = self.config.dyn[self.config.dyn.typ].context
        carry = carry + (jnp.zeros((B, m), f32),)
      else:
        S = self.config.dyn[self.config.dyn.typ].stoch
        C = self.config.dyn[self.config.dyn.typ].classes
        carry = carry + (jnp.zeros((B, S, C), f32),)
    return carry, metrics

  def _apply_replay_context(self, carry, data):
    if self.config.thick.enabled:
      (enc_carry, dyn_carry, dec_carry, prevact, _goal) = carry
    else:
      (enc_carry, dyn_carry, dec_carry, prevact) = carry
    carry = (enc_carry, dyn_carry, dec_carry)
    stepid = data['stepid']
    obs = {k: data[k] for k in self.obs_space}
    prepend = lambda x, y: jnp.concatenate([x[:, None], y[:, :-1]], 1)
    prevact = {k: prepend(prevact[k], data[k]) for k in self.act_space}
    if not self.config.replay_context:
      return carry, obs, prevact, stepid

    K = self.config.replay_context
    nested = elements.tree.nestdict(data)
    entries = [nested.get(k, {}) for k in ('enc', 'dyn', 'dec')]
    lhs = lambda xs: jax.tree.map(lambda x: x[:, :K], xs)
    rhs = lambda xs: jax.tree.map(lambda x: x[:, K:], xs)
    rep_carry = (
        self.enc.truncate(lhs(entries[0]), enc_carry),
        self.dyn.truncate(lhs(entries[1]), dyn_carry),
        self.dec.truncate(lhs(entries[2]), dec_carry))
    rep_obs = {k: rhs(data[k]) for k in self.obs_space}
    rep_prevact = {k: data[k][:, K - 1: -1] for k in self.act_space}
    rep_stepid = rhs(stepid)

    first_chunk = (data['consec'][:, 0] == 0)
    carry, obs, prevact, stepid = jax.tree.map(
        lambda normal, replay: nn.where(first_chunk, replay, normal),
        (carry, rhs(obs), rhs(prevact), rhs(stepid)),
        (rep_carry, rep_obs, rep_prevact, rep_stepid))
    return carry, obs, prevact, stepid

  def _make_opt(
      self,
      lr: float = 4e-5,
      agc: float = 0.3,
      eps: float = 1e-20,
      beta1: float = 0.9,
      beta2: float = 0.999,
      momentum: bool = True,
      nesterov: bool = False,
      wd: float = 0.0,
      wdregex: str = r'/kernel$',
      schedule: str = 'const',
      warmup: int = 1000,
      anneal: int = 0,
  ):
    chain = []
    chain.append(embodied.jax.opt.clip_by_agc(agc))
    chain.append(embodied.jax.opt.scale_by_rms(beta2, eps))
    chain.append(embodied.jax.opt.scale_by_momentum(beta1, nesterov))
    if wd:
      assert not wdregex[0].isnumeric(), wdregex
      pattern = re.compile(wdregex)
      wdmask = lambda params: {k: bool(pattern.search(k)) for k in params}
      chain.append(optax.add_decayed_weights(wd, wdmask))
    assert anneal > 0 or schedule == 'const'
    if schedule == 'const':
      sched = optax.constant_schedule(lr)
    elif schedule == 'linear':
      sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
    elif schedule == 'cosine':
      sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
    else:
      raise NotImplementedError(schedule)
    if warmup:
      ramp = optax.linear_schedule(0.0, lr, warmup)
      sched = optax.join_schedules([ramp, sched], [warmup])
    chain.append(optax.scale_by_learning_rate(sched))
    return optax.chain(*chain)


def imag_loss(
    act, rew, con,
    policy, value, slowvalue,
    retnorm, valnorm, advnorm,
    update,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
):
  losses = {}
  metrics = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 if contdisc else 1 - 1 / horizon
  weight = jnp.cumprod(disc * con, 1) / disc
  last = jnp.zeros_like(con)
  term = 1 - con
  ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)

  baseline = tarval[:, :-1]

  metrics['val_mae'] = jnp.abs(val[:, :-1] - ret).mean()

  roffset, rscale = retnorm(ret, update)
  adv = (ret - baseline) / rscale
  aoffset, ascale = advnorm(adv, update)
  adv_normed = (adv - aoffset) / ascale
  logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
  ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
  policy_loss = sg(weight[:, :-1]) * -(
      logpi * sg(adv_normed) + actent * sum(ents.values()))
  losses['policy'] = policy_loss

  # Critic loss
  voffset, vscale = valnorm(ret, update)
  tar_normed = (ret - voffset) / vscale
  tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)
  losses['value'] = sg(weight[:, :-1]) * (
      value.loss(sg(tar_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  ret_normed = (ret - roffset) / rscale
  metrics['adv'] = adv.mean()
  metrics['adv_std'] = adv.std()
  metrics['adv_mag'] = jnp.abs(adv).mean()
  metrics['rew'] = rew.mean()
  metrics['con'] = con.mean()
  metrics['ret'] = ret_normed.mean()
  metrics['val'] = val.mean()
  metrics['tar'] = tar_normed.mean()
  metrics['weight'] = weight.mean()
  metrics['slowval'] = slowval.mean()
  metrics['ret_min'] = ret_normed.min()
  metrics['ret_max'] = ret_normed.max()
  metrics['ret_rate'] = (jnp.abs(ret_normed) >= 1.0).mean()
  for k in act:
    metrics[f'ent/{k}'] = ents[k].mean()
    if hasattr(policy[k], 'minent'):
      lo, hi = policy[k].minent, policy[k].maxent
      metrics[f'rand/{k}'] = (ents[k].mean() - lo) / (hi - lo)

  outs = {}
  outs['ret'] = ret
  return losses, outs, metrics


def mgr_imag_loss(
    act, rew, logits,
    value, slowvalue,
    retnorm, valnorm, advnorm,
    hlwm, update, horizon, actent, lam, slowreg,
):
  """Manager actor-critic on coarse imagination trajectory."""
  losses = {}
  metrics = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval  # always use slow target

  disc = 1 - 1 / horizon
  con = jnp.ones_like(rew)  # no termination in coarse imagination
  weight = jnp.cumprod(disc * con, 1) / disc
  last = jnp.zeros_like(rew)
  term = jnp.zeros_like(rew)
  ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)

  baseline = tarval[:, :-1]
  roffset, rscale = retnorm(ret, update)
  adv = (ret - baseline) / rscale
  aoffset, ascale = advnorm(adv, update)
  adv_normed = (adv - aoffset) / ascale

  logpi = hlwm._hl_act_logp(logits[:, :-1], act[:, :-1])
  ent = hlwm._hl_act_entropy(logits[:, :-1])

  losses['mgr_policy'] = sg(weight[:, :-1]) * -(
      logpi * sg(adv_normed) + actent * ent)

  # Coarse critic loss
  voffset, vscale = valnorm(ret, update)
  tar_normed = (ret - voffset) / vscale
  tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)
  losses['coarse_val'] = sg(weight[:, :-1]) * (
      value.loss(sg(tar_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  metrics['mgr_val'] = val.mean()
  ret_normed = (ret - roffset) / rscale
  metrics['mgr_ret'] = ret_normed.mean()
  metrics['mgr_adv'] = adv.mean()
  metrics['mgr_ent'] = ent.mean()
  metrics['mgr_val_mae'] = jnp.abs(val[:, :-1] - ret).mean()

  return losses, metrics


def repl_loss(
    last, term, rew, boot,
    value, slowvalue, valnorm,
    update=True,
    slowreg=1.0,
    slowtar=True,
    horizon=333,
    lam=0.95,
):
  losses = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 - 1 / horizon
  weight = f32(~last)
  ret = lambda_return(last, term, rew, tarval, boot, disc, lam)

  voffset, vscale = valnorm(ret, update)
  ret_normed = (ret - voffset) / vscale
  ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
  losses['repval'] = weight[:, :-1] * (
      value.loss(sg(ret_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  outs = {}
  outs['ret'] = ret
  metrics = {}

  return losses, outs, metrics


def lambda_return(last, term, rew, val, boot, disc, lam):
  chex.assert_equal_shape((last, term, rew, val, boot))
  rets = [boot[:, -1]]
  live = (1 - f32(term))[:, 1:] * disc
  cont = (1 - f32(last))[:, 1:] * lam
  interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
  for t in reversed(range(live.shape[1])):
    rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
  return jnp.stack(list(reversed(rets))[:-1], 1)
