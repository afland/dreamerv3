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
    self.dyn = {
        'rssm': rssm.RSSM,
        'crssm': crssm.CRSSM,
    }[config.dyn.typ](act_space, **config.dyn[config.dyn.typ], name='dyn')
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
      # Coarse pathway: context only
      self.coarse_feat2tensor = lambda x: nn.cast(x['context'])
    else:
      self.feat2tensor = lambda x: jnp.concatenate([
          nn.cast(x['deter']),
          nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1)))], -1)
      self.coarse_feat2tensor = None

    if config.thick.goal_in_policy:
      assert config.thick.enabled, 'goal_in_policy requires thick.enabled'
      _base = self.feat2tensor
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
          hl_act_dim=config.thick.hl_act_dim,
          **config.thick.hlwm, name='hlwm')
      self.coarse_val = embodied.jax.MLPHead(
          scalar, **config.value, name='coarse_val')
      self.slow_coarse_val = embodied.jax.SlowModel(
          embodied.jax.MLPHead(scalar, **config.value, name='slow_coarse_val'),
          source=self.coarse_val, **config.slowvalue)
    else:
      self.hlwm = None
      self.coarse_val = None
      self.slow_coarse_val = None

    self.modules = [
        self.dyn, self.enc, self.dec, self.rew, self.con, self.pol, self.val]
    if self.coarse_rew:
      self.modules.extend([self.coarse_rew, self.coarse_con])
      if config.loss_scales.get('coarse_rec', 0.0) != 0.0:
        self.modules.append(self.coarse_dec)
    if self.hlwm:
      self.modules.extend([self.hlwm, self.coarse_val])

    self.opt = embodied.jax.Optimizer(
        self.modules, self._make_opt(**config.opt), summary_depth=1,
        name='opt')

    scales = self.config.loss_scales.copy()
    rec = scales.pop('rec')
    scales.update({k: rec for k in dec_space})
    # Remove scales not needed for current config
    if config.dyn.typ != 'crssm':
      for k in ('coarse_dyn', 'sparse', 'gate_info',
                'coarse_rec', 'coarse_rew', 'coarse_con', 'gate_improve'):
        scales.pop(k, None)
    if not config.thick.enabled:
      for k in ('hlwm_stoch', 'hlwm_action', 'hlwm_time',
                'hlwm_reward', 'hlwm_act_kl', 'coarse_val'):
        scales.pop(k, None)
    self.scales = {k: v for k, v in scales.items() if float(v) != 0.0}

  @staticmethod
  def _cosine_sim(logit_a, logit_b):
    a = logit_a.reshape((*logit_a.shape[:-2], -1))
    b = logit_b.reshape((*logit_b.shape[:-2], -1))
    return (a * b).sum(-1) / (jnp.linalg.norm(a, -1) * jnp.linalg.norm(b, -1) + 1e-8)

  def _imagine_with_goals(self, starts, z_goals, H, training):
    """Custom imagination loop that threads goals through policy input.

    Goals advance to the next tree-search subgoal whenever a boundary gate fires.
    """
    K_plan = self.config.thick.plan_depth
    BK = starts['deter'].shape[0]

    def step(carry, _):
      crssm_carry, goal_idx, goals = carry
      current_goal = goals[jnp.arange(BK), goal_idx]
      feat_for_pol = {**sg(crssm_carry), 'goal': current_goal}
      action = sample(self.pol(self.pol_feat2tensor(feat_for_pol), 1))
      crssm_carry, (feat, act) = self.dyn.imagine(
          crssm_carry, action, 1, training, single=True)
      gate_bin = feat['gate_binary']
      goal_idx = jnp.where(gate_bin > 0.5,
                           jnp.minimum(goal_idx + 1, K_plan - 1), goal_idx)
      current_goal = goals[jnp.arange(BK), goal_idx]
      feat = {**feat, 'goal': current_goal}
      return (crssm_carry, goal_idx, goals), (feat, act)

    init = (nn.cast(starts), jnp.zeros(BK, i32), z_goals)
    final, (imgfeat, imgact) = nj.scan(step, init, (), H, unroll=1, axis=1)
    return final[0], imgfeat, imgact

  def _plan_tree_search(self, context, stoch, training):
    """Exhaustive tree search over HL action space.

    Args:
      context: [BK, m] current context
      stoch: [BK, S, C] current stochastic state

    Returns:
      z_goals: [BK, K, S, C] stoch logits from best plan at each HL step
      metrics: dict of scalar metrics
    """
    D = self.config.thick.hl_act_dim
    K = self.config.thick.plan_depth
    N = D ** K  # total sequences
    BK = context.shape[0]
    S = stoch.shape[-2]
    C = stoch.shape[-1]
    disc = 1 - 1 / self.config.horizon
    metrics = {}

    # Generate all D^K action index sequences: [N, K]
    # Each row is a sequence of K action indices in [0, D)
    act_indices = jnp.array(
        np.array(np.meshgrid(*[np.arange(D)] * K)).T.reshape(-1, K))

    # Tile inputs: [BK, ...] -> [BK*N, ...]
    tile = lambda x: jnp.repeat(x, N, axis=0)
    c = tile(context)       # [BK*N, m]
    z = tile(stoch)         # [BK*N, S, C]

    # Tile action sequences: [N, K] -> [BK*N, K]
    act_seq = jnp.tile(act_indices, (BK, 1))  # [BK*N, K]

    total_return = jnp.zeros(BK * N)
    cumul_dt = jnp.zeros(BK * N)
    all_stoch_logits = []

    for k in range(K):
      # One-hot encode action for this step
      hl_act = jax.nn.one_hot(act_seq[:, k], D)  # [BK*N, D]
      hl_act = nn.cast(hl_act)

      # HLWM predict given this action
      preds = self.hlwm.predict_given_action(hl_act, c, z)
      pred_stoch_flat = nn.cast(preds['stoch'])      # [BK*N, S*C]
      pred_action = nn.cast(preds['action'])          # [BK*N, A]
      stoch_logit = preds['stoch_logit']              # [BK*N, S, C]

      # Accumulate discounted reward
      gamma_dt = jnp.power(disc, cumul_dt)
      total_return = total_return + gamma_dt * preds['reward']
      cumul_dt = cumul_dt + preds['time_delta']

      all_stoch_logits.append(stoch_logit)

      # Step coarse dynamics for next HL step (if not last)
      if k < K - 1:
        c = self.dyn.context_step(c, pred_stoch_flat, pred_action)
        z_logit = self.dyn._coarse_prior(c, sg(pred_stoch_flat), sg(pred_action))
        z = nn.cast(self.dyn._dist(z_logit).sample(seed=nj.seed()))

    # Optionally bootstrap leaf with coarse critic
    if self.config.thick.use_coarse_critic:
      # Final context step to get leaf state
      c = self.dyn.context_step(c, pred_stoch_flat, pred_action)
      z_logit = self.dyn._coarse_prior(c, sg(pred_stoch_flat), sg(pred_action))
      z_leaf = nn.cast(self.dyn._dist(z_logit).sample(seed=nj.seed()))
      z_leaf_flat = z_leaf.reshape((*z_leaf.shape[:-2], -1))
      leaf_inp = jnp.concatenate([c, z_leaf_flat], -1)
      leaf_val = self.coarse_val(leaf_inp, 1).pred()
      gamma_dt = jnp.power(disc, cumul_dt)
      total_return = total_return + gamma_dt * leaf_val

    # Reshape to [BK, N], find best plan
    total_return = total_return.reshape(BK, N)
    best_idx = jnp.argmax(total_return, axis=1)  # [BK]

    # Stack stoch logits: [K, BK*N, S, C] -> gather best
    all_logits = jnp.stack(all_stoch_logits, axis=0)  # [K, BK*N, S, C]
    all_logits = all_logits.reshape(K, BK, N, S, C)
    # Gather best plan's logits for each BK
    bk_idx = jnp.arange(BK)
    z_goals = all_logits[:, bk_idx, best_idx]  # [K, BK, S, C]
    z_goals = jnp.moveaxis(z_goals, 0, 1)      # [BK, K, S, C]

    metrics['plan/best_return'] = total_return[bk_idx, best_idx].mean()
    metrics['plan/mean_return'] = total_return.mean()
    metrics['plan/return_std'] = total_return.std(axis=1).mean()

    return sg(z_goals), metrics

  @property
  def policy_keys(self):
    if self.config.thick.goal_in_policy:
      return '^(enc|dyn|dec|pol|hlwm)/'
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
    if self.config.thick.goal_in_policy:
      S = self.config.dyn[self.config.dyn.typ].stoch
      C = self.config.dyn[self.config.dyn.typ].classes
      carry = carry + (jnp.zeros((batch_size, S, C), f32),)
    return carry

  def init_train(self, batch_size):
    return self.init_policy(batch_size)

  def init_report(self, batch_size):
    return self.init_policy(batch_size)

  def policy(self, carry, obs, mode='train'):
    if self.config.thick.goal_in_policy:
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
    if self.config.thick.goal_in_policy and self.hlwm:
      # Replan on gate fire: run tree search, take first goal
      gate_fired = feat['gate_binary'] > 0.5  # [B]
      z_goals, _ = self._plan_tree_search(
          feat['context'], feat['stoch'], training=False)
      new_goal = z_goals[:, 0]  # [B, S, C]
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
    if self.config.thick.goal_in_policy:
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
      improvement = jnp.zeros_like(gate_prob)
      # Coarse dyn: KL(posterior || coarse_prior) with old vs new context
      post = repfeat['logit']
      z_flat = sg(repfeat['stoch'].reshape((*repfeat['stoch'].shape[:-2], -1)))
      actemb = nn.DictConcat(self.act_space, 1)(prevact)
      actemb /= sg(jnp.maximum(1, jnp.abs(actemb)))
      logit_old = self.dyn._coarse_prior(ctx_old, z_flat, actemb)
      logit_new = self.dyn._coarse_prior(ctx_new, z_flat, actemb)
      improvement += (self.dyn._dist(sg(post)).kl(self.dyn._dist(logit_old))
                      - self.dyn._dist(sg(post)).kl(self.dyn._dist(logit_new)))
      # Coarse rew/con: prediction loss with old vs new context
      if 'coarse_rew' in self.scales:
        inp_old = nn.cast(ctx_old)
        inp_new = nn.cast(ctx_new)
        improvement += (self.coarse_rew(inp_old, 2).loss(obs['reward'])
                        - self.coarse_rew(inp_new, 2).loss(obs['reward']))
      if 'coarse_con' in self.scales:
        inp_old = nn.cast(ctx_old)
        inp_new = nn.cast(ctx_new)
        improvement += (self.coarse_con(inp_old, 2).loss(con)
                        - self.coarse_con(inp_new, 2).loss(con))
      # Coarse rec: reconstruction loss with old vs new context
      if 'coarse_rec' in self.scales:
        inp_old = nn.cast(ctx_old)
        inp_new = nn.cast(ctx_new)
        improvement += (sum(self.coarse_dec(inp_old, 2, obs).values())
                        - sum(self.coarse_dec(inp_new, 2, obs).values()))
      if 'gate_improve' in self.scales:
        losses['gate_improve'] = -gate_prob * sg(jax.nn.relu(improvement))
      metrics['gate_improve_mean'] = improvement.mean()
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
    if self.config.thick.goal_in_policy and self.hlwm:
      # Goal-in-policy: run tree search FIRST, then imagine with goals
      hlwm_mask = f32(self.opt.step.read() >= self.config.thick.hlwm_start)
      first = jax.tree.map(
          lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])), repfeat)
      starts_ctx = starts['context']   # [BK, m]
      starts_z = starts['stoch']       # [BK, S, C]
      z_goals, plan_mets = self._plan_tree_search(starts_ctx, starts_z, training)

      # Imagine with goal-conditioned policy
      _, imgfeat, imgprevact = self._imagine_with_goals(
          starts, z_goals, H, training)

      # Prepend first timestep with goal from z_goals[:, 0]
      first = {**sg(first, skip=self.config.ac_grads),
               'goal': z_goals[:, 0:1]}
      imgfeat = concat([first, sg(imgfeat)], 1)

      # Last action uses goal-conditioned policy
      last_feat = jax.tree.map(lambda x: x[:, -1], imgfeat)
      lastact = sample(self.pol(self.pol_feat2tensor(last_feat), 1))
      lastact = jax.tree.map(lambda x: x[:, None], lastact)
      imgact = concat([imgprevact, lastact], 1)

      # Cosine sim reward bonus uses goals already in imgfeat
      sim = self._cosine_sim(imgfeat['logit'], sg(imgfeat['goal']))
      inp = self.feat2tensor(imgfeat)
      rew = self.rew(inp, 2).pred()
      rew = rew + self.config.thick.kappa * hlwm_mask * sim
      plan_mets['plan/sim_mean'] = sim.mean()
      plan_mets['plan/rew_bonus'] = (self.config.thick.kappa * sim).mean()
    else:
      # Default path: imagine then optionally run tree search
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

      # Tree search planning + z_goal reward augmentation (THICK)
      if self.hlwm:
        hlwm_mask = f32(self.opt.step.read() >= self.config.thick.hlwm_start)
        starts_ctx = imgfeat['context'][:, 0]   # [BK, m]
        starts_z = imgfeat['stoch'][:, 0]       # [BK, S, C]
        z_goals, plan_mets = self._plan_tree_search(starts_ctx, starts_z, training)
        # Forward scan: assign z_goal per timestep, advancing at boundaries
        gate_bin = imgfeat['gate_binary']  # [BK, H+1]
        BK_ = gate_bin.shape[0]
        K_ = self.config.thick.plan_depth
        def assign_goals(carry, gate_t):
          idx, goals = carry
          idx = jnp.where(gate_t > 0.5, jnp.minimum(idx + 1, K_ - 1), idx)
          return (idx, goals), goals[jnp.arange(BK_), idx]
        _, z_goal_per_t = jax.lax.scan(
            assign_goals, (jnp.zeros(BK_, i32), z_goals),
            jnp.moveaxis(gate_bin, 1, 0))
        z_goal_per_t = jnp.moveaxis(z_goal_per_t, 0, 1)  # [BK, H+1, S, C]
        # Augment reward with cosine similarity toward z_goal
        sim = self._cosine_sim(imgfeat['logit'], sg(z_goal_per_t))
        rew = rew + self.config.thick.kappa * hlwm_mask * sim
        plan_mets['plan/sim_mean'] = sim.mean()
        plan_mets['plan/rew_bonus'] = (self.config.thick.kappa * sim).mean()
    metrics.update(plan_mets)

    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgfeat))
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgact))

    # Policy input: use pol_feat2tensor (includes goal when goal_in_policy)
    pol_inp = self.pol_feat2tensor(imgfeat)

    los, imgloss_out, mets = imag_loss(
        imgact, rew,
        self.con(inp, 2).prob(1),
        self.pol(pol_inp, 2),
        self.val(inp, 2),
        self.slowval(inp, 2),
        self.retnorm, self.valnorm, self.advnorm,
        update=training,
        contdisc=self.config.contdisc,
        horizon=self.config.horizon,
        coarse_value=self.coarse_val(self._coarse_critic_inp(imgfeat), 2) if self.hlwm else None,
        slow_coarse_value=self.slow_coarse_val(self._coarse_critic_inp(imgfeat), 2) if self.hlwm else None,
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

    carry = (*new_carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, metrics

  def _apply_replay_context(self, carry, data):
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
    coarse_value=None,
    slow_coarse_value=None,
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
  if coarse_value is not None:
    coarse_val_pred = coarse_value.pred() * vscale + voffset
    metrics['coarse_val_mae'] = jnp.abs(coarse_val_pred[:, :-1] - ret).mean()
    diff = val[:, :-1] - coarse_val_pred[:, :-1]
    metrics['critic_diff_abs'] = jnp.abs(diff).mean()
    metrics['critic_diff'] = diff.mean()

  roffset, rscale = retnorm(ret, update)
  adv = (ret - baseline) / rscale
  aoffset, ascale = advnorm(adv, update)
  adv_normed = (adv - aoffset) / ascale
  logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
  ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
  policy_loss = sg(weight[:, :-1]) * -(
      logpi * sg(adv_normed) + actent * sum(ents.values()))
  losses['policy'] = policy_loss

  # Critic losses: both critics regress V_lambda
  voffset, vscale = valnorm(ret, update)
  tar_normed = (ret - voffset) / vscale
  tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)
  losses['value'] = sg(weight[:, :-1]) * (
      value.loss(sg(tar_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]
  if coarse_value is not None:
    losses['coarse_val'] = sg(weight[:, :-1]) * (
        coarse_value.loss(sg(tar_padded)) +
        slowreg * coarse_value.loss(sg(slow_coarse_value.pred())))[:, :-1]

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
