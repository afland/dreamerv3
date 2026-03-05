import math

import einops
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from . import gatelord

f32 = jnp.float32
sg = jax.lax.stop_gradient


class CRSSM(nj.Module):
  """Context-augmented RSSM (C-RSSM) with GateL0RD context memory.

  Adds a slowly-changing context vector updated by GateL0RD gates,
  plus a coarse prior pathway independent of the deterministic state.
  """

  deter: int = 4096
  hidden: int = 1024
  stoch: int = 32
  classes: int = 32
  context: int = 16
  gate_type: str = 'gatelord'
  gate_noise_scale: float = 0.1
  gate_noise_always: bool = True
  norm: str = 'rms'
  act: str = 'silu'
  unroll: bool = False
  unimix: float = 0.01
  outscale: float = 1.0
  imglayers: int = 2
  obslayers: int = 1
  dynlayers: int = 1
  coarse_layers: int = 1
  absolute: bool = False
  blocks: int = 8
  free_nats: float = 1.0
  sparse_free: float = 0.0

  hide_context: bool = False

  def __init__(self, act_space, **kw):
    assert self.deter % self.blocks == 0
    self.act_space = act_space
    kw.pop('hide_context', None)
    kw.pop('gate_noise_always', None)
    self.kw = kw

  @property
  def entry_space(self):
    return dict(
        deter=elements.Space(np.float32, self.deter),
        stoch=elements.Space(np.float32, (self.stoch, self.classes)),
        context=elements.Space(np.float32, self.context))

  def initial(self, bsize):
    carry = nn.cast(dict(
        deter=jnp.zeros([bsize, self.deter], f32),
        stoch=jnp.zeros([bsize, self.stoch, self.classes], f32),
        context=jnp.zeros([bsize, self.context], f32)))
    return carry

  def truncate(self, entries, carry=None):
    assert entries['deter'].ndim == 3, entries['deter'].shape
    carry = jax.tree.map(lambda x: x[:, -1], entries)
    return carry

  def starts(self, entries, carry, nlast):
    B = len(jax.tree.leaves(carry)[0])
    return jax.tree.map(
        lambda x: x[:, -nlast:].reshape((B * nlast, *x.shape[2:])), entries)

  def observe(self, carry, tokens, action, reset, training, single=False):
    carry, tokens, action = nn.cast((carry, tokens, action))
    if single:
      carry, (entry, feat) = self._observe(
          carry, tokens, action, reset, training)
      return carry, entry, feat
    else:
      unroll = jax.tree.leaves(tokens)[0].shape[1] if self.unroll else 1
      carry, (entries, feat) = nj.scan(
          lambda carry, inputs: self._observe(
              carry, *inputs, training),
          carry, (tokens, action, reset), unroll=unroll, axis=1)
      return carry, entries, feat

  def _observe(self, carry, tokens, action, reset, training):
    # 1. Mask carry + action on reset (including context)
    deter, stoch, ctx, action = nn.mask(
        (carry['deter'], carry['stoch'], carry['context'], action), ~reset)
    action = nn.DictConcat(self.act_space, 1)(action)
    action = nn.mask(action, ~reset)
    action /= sg(jnp.maximum(1, jnp.abs(action)))

    # 2. Shared projections for stoch and action
    stoch_flat = stoch.reshape((stoch.shape[0], -1))
    stoch_proj = self.sub('stoch_proj', nn.Linear, self.hidden, **self.kw)(stoch_flat)
    stoch_proj = nn.act(self.act)(self.sub('stoch_proj_norm', nn.Norm, self.norm)(stoch_proj))
    action_proj = self.sub('action_proj', nn.Linear, self.hidden, **self.kw)(action)
    action_proj = nn.act(self.act)(self.sub('action_proj_norm', nn.Norm, self.norm)(action_proj))

    # 3. Gate cell: update context (Eq. 2)
    gate_input = jnp.concatenate([stoch_proj, action_proj], -1)
    coarse_feat, ctx, gates = self._gate_cell()(gate_input, ctx, training or self.gate_noise_always)

    # 4. GRU core with context (Eq. 3)
    deter = self._core(deter, stoch_proj, action_proj, ctx)

    # 5. Posterior from [deter, context, tokens] (Eq. 6)
    tokens = tokens.reshape((*deter.shape[:-1], -1))
    x = tokens if self.absolute else jnp.concatenate([deter, ctx, tokens], -1)
    for i in range(self.obslayers):
      x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
    logit = self._logit('obslogit', x)
    stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))

    # 6. Coarse prior from output-gated feature + prev stoch/action (Eq. 5)
    coarse_logit = self._coarse_prior(coarse_feat, stoch_flat, action)

    carry = dict(deter=deter, stoch=stoch, context=ctx)
    feat = dict(deter=deter, stoch=stoch, logit=logit, context=ctx,
                coarse_logit=coarse_logit, gates=gates)
    entry = dict(deter=deter, stoch=stoch, context=ctx)
    assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
    return carry, (entry, feat)

  def imagine(self, carry, policy, length, training, single=False):
    if single:
      action = policy(sg(carry)) if callable(policy) else policy
      actemb = nn.DictConcat(self.act_space, 1)(action)

      # Shared projections
      stoch_flat = carry['stoch'].reshape((carry['stoch'].shape[0], -1))
      stoch_proj = self.sub('stoch_proj', nn.Linear, self.hidden, **self.kw)(stoch_flat)
      stoch_proj = nn.act(self.act)(self.sub('stoch_proj_norm', nn.Norm, self.norm)(stoch_proj))
      action_proj = self.sub('action_proj', nn.Linear, self.hidden, **self.kw)(actemb)
      action_proj = nn.act(self.act)(self.sub('action_proj_norm', nn.Norm, self.norm)(action_proj))

      # Gate cell
      gate_input = jnp.concatenate([stoch_proj, action_proj], -1)
      coarse_feat, ctx, gates = self._gate_cell()(gate_input, carry['context'], training or self.gate_noise_always)

      # GRU core
      deter = self._core(carry['deter'], stoch_proj, action_proj, ctx)

      # Precise prior (Eq. 4)
      logit = self._prior(deter, ctx)
      stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))

      # Coarse prior (Eq. 5)
      coarse_logit = self._coarse_prior(coarse_feat, stoch_flat, actemb)

      carry = nn.cast(dict(deter=deter, stoch=stoch, context=ctx))
      feat = nn.cast(dict(deter=deter, stoch=stoch, logit=logit, context=ctx,
                          coarse_logit=coarse_logit, gates=gates))
      assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
      return carry, (feat, action)
    else:
      unroll = length if self.unroll else 1
      if callable(policy):
        carry, (feat, action) = nj.scan(
            lambda c, _: self.imagine(c, policy, 1, training, single=True),
            nn.cast(carry), (), length, unroll=unroll, axis=1)
      else:
        carry, (feat, action) = nj.scan(
            lambda c, a: self.imagine(c, a, 1, training, single=True),
            nn.cast(carry), nn.cast(policy), length, unroll=unroll, axis=1)
      return carry, feat, action

  def loss(self, carry, tokens, acts, reset, training):
    metrics = {}
    carry, entries, feat = self.observe(carry, tokens, acts, reset, training)

    # Precise prior KL (Eq. 4)
    prior = self._prior(feat['deter'], feat['context'])
    post = feat['logit']
    dyn = self._dist(sg(post)).kl(self._dist(prior))
    rep = self._dist(post).kl(self._dist(sg(prior)))
    if self.free_nats:
      dyn = jnp.maximum(dyn, self.free_nats)
      rep = jnp.maximum(rep, self.free_nats)

    # Coarse prior KL (Eq. 5)
    coarse_prior = feat['coarse_logit']
    coarse_dyn = self._dist(sg(post)).kl(self._dist(coarse_prior))
    coarse_rep = self._dist(post).kl(self._dist(sg(coarse_prior)))
    if self.free_nats:
      coarse_dyn = jnp.maximum(coarse_dyn, self.free_nats)
      coarse_rep = jnp.maximum(coarse_rep, self.free_nats)

    # Sparsity loss (mean for gatelord, budget for timelord)
    sparse_mode = 'budget' if self.gate_type == 'timelord' else 'mean'
    sparse = gatelord.sparsity_loss(
        feat['gates'], free=self.sparse_free, mode=sparse_mode)

    losses = {
        'dyn': dyn, 'rep': rep,
        'coarse_dyn': coarse_dyn, 'coarse_rep': coarse_rep,
        'sparse': sparse,
    }
    metrics['dyn_ent'] = self._dist(prior).entropy().mean()
    metrics['rep_ent'] = self._dist(post).entropy().mean()
    metrics['coarse_ent'] = self._dist(coarse_prior).entropy().mean()
    # Per-timestep change frequency: fraction of steps where any gate > 0
    gate_open = f32(feat['gates'].sum(-1) > 0)  # [B, T]
    metrics['gate_change_freq'] = gate_open.mean()
    if self.gate_type == 'gatelord':
      # Mean magnitude of gates when open (GateLord has soft gates)
      gate_flat = feat['gates'].reshape(-1, feat['gates'].shape[-1])
      open_mask = gate_flat.sum(-1) > 0
      open_mag = jnp.where(open_mask[:, None], gate_flat, 0.0).sum()
      open_count = jnp.maximum(open_mask.sum() * feat['gates'].shape[-1], 1)
      metrics['gate_magnitude'] = open_mag / open_count
    return carry, entries, losses, feat, metrics

  def _gate_cell(self):
    cells = {'gatelord': gatelord.GateLord, 'timelord': gatelord.TimeLord}
    cls = cells[self.gate_type]
    kw = dict(noise_scale=self.gate_noise_scale, **self.kw)
    return self.sub('gate_cell', cls, self.context, **kw)

  def _core(self, deter, stoch_proj, action_proj, context):
    """GRU core with 4 inputs: deter, stoch_proj, action_proj, context."""
    g = self.blocks
    flat2group = lambda x: einops.rearrange(x, '... (g h) -> ... g h', g=g)
    group2flat = lambda x: einops.rearrange(x, '... g h -> ... (g h)', g=g)

    # Project each input (stoch/action already projected, reuse those)
    x0 = self.sub('dynin0', nn.Linear, self.hidden, **self.kw)(deter)
    x0 = nn.act(self.act)(self.sub('dynin0norm', nn.Norm, self.norm)(x0))
    # stoch_proj and action_proj are already projected
    x3 = self.sub('dynin3', nn.Linear, self.hidden, **self.kw)(context)
    x3 = nn.act(self.act)(self.sub('dynin3norm', nn.Norm, self.norm)(x3))

    x = jnp.concatenate([x0, stoch_proj, action_proj, x3], -1)[..., None, :].repeat(g, -2)
    x = group2flat(jnp.concatenate([flat2group(deter), x], -1))
    for i in range(self.dynlayers):
      x = self.sub(f'dynhid{i}', nn.BlockLinear, self.deter, g, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'dynhid{i}norm', nn.Norm, self.norm)(x))
    x = self.sub('dyngru', nn.BlockLinear, 3 * self.deter, g, **self.kw)(x)
    gates = jnp.split(flat2group(x), 3, -1)
    reset, cand, update = [group2flat(x) for x in gates]
    reset = jax.nn.sigmoid(reset)
    cand = jnp.tanh(reset * cand)
    update = jax.nn.sigmoid(update - 1)
    deter = update * cand + (1 - update) * deter
    return deter

  def _prior(self, deter, context):
    """Precise prior (Eq. 4): MLP from [deter, context]."""
    x = jnp.concatenate([deter, context], -1)
    for i in range(self.imglayers):
      x = self.sub(f'prior{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'prior{i}norm', nn.Norm, self.norm)(x))
    return self._logit('priorlogit', x)

  def _coarse_prior(self, coarse_feat, prev_stoch_flat, prev_action):
    """Coarse prior (Eq. 5): MLP from output-gated feature + prev stoch/action.

    Independent of h_t, creating the coarse processing pathway.
    """
    x = jnp.concatenate([coarse_feat, prev_stoch_flat, prev_action], -1)
    for i in range(self.coarse_layers):
      x = self.sub(f'coarse{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'coarse{i}norm', nn.Norm, self.norm)(x))
    return self._logit('coarselogit', x)

  def _logit(self, name, x):
    kw = dict(**self.kw, outscale=self.outscale)
    x = self.sub(name, nn.Linear, self.stoch * self.classes, **kw)(x)
    return x.reshape(x.shape[:-1] + (self.stoch, self.classes))

  def _dist(self, logits):
    out = embodied.jax.outs.OneHot(logits, self.unimix)
    out = embodied.jax.outs.Agg(out, 1, jnp.sum)
    return out
