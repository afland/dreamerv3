import math

import einops
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

f32 = jnp.float32
sg = jax.lax.stop_gradient


class CRSSM(nj.Module):
  """Context-augmented RSSM (C-RSSM) with BlockGRU context memory.

  Adds a slowly-changing context vector updated by a BlockGRU wrapped with
  a scalar binary boundary gate. Context never enters the fine pathway.
  Coarse prior takes only context.
  """

  deter: int = 4096
  hidden: int = 1024
  coarse_hidden: int = 512
  stoch: int = 32
  classes: int = 32
  context: int = 4096
  boundary_prior: float = 0.1
  min_gate_rate: float = 0.05
  stochastic_gate: bool = True
  norm: str = 'rms'
  act: str = 'silu'
  unroll: bool = False
  unimix: float = 0.01
  outscale: float = 1.0
  imglayers: int = 2
  obslayers: int = 1
  dynlayers: int = 1
  absolute: bool = False
  blocks: int = 8
  free_nats: float = 1.0

  def __init__(self, act_space, **kw):
    assert self.deter % self.blocks == 0
    assert self.context % self.blocks == 0
    self.act_space = act_space
    self.kw = kw

  @property
  def entry_space(self):
    return dict(
        deter=elements.Space(np.float32, self.deter),
        stoch=elements.Space(np.float32, (self.stoch, self.classes)),
        context=elements.Space(np.float32, self.context),
        prev_reset=elements.Space(np.float32))

  def initial(self, bsize):
    carry = nn.cast(dict(
        deter=jnp.zeros([bsize, self.deter], f32),
        stoch=jnp.zeros([bsize, self.stoch, self.classes], f32),
        context=jnp.zeros([bsize, self.context], f32),
        prev_reset=jnp.ones([bsize], f32)))
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
    prev_reset = carry.get('prev_reset', jnp.zeros_like(reset, dtype=f32))
    deter, stoch, ctx, action = nn.mask(
        (carry['deter'], carry['stoch'], carry['context'], action), ~reset)
    action = nn.DictConcat(self.act_space, 1)(action)
    action = nn.mask(action, ~reset)
    action /= sg(jnp.maximum(1, jnp.abs(action)))

    # 2. Fine projections for stoch and action (fine pathway)
    stoch_flat = stoch.reshape((stoch.shape[0], -1))
    stoch_proj = self.sub('stoch_proj', nn.Linear, self.hidden, **self.kw)(stoch_flat)
    stoch_proj = nn.act(self.act)(self.sub('stoch_proj_norm', nn.Norm, self.norm)(stoch_proj))
    action_proj = self.sub('action_proj', nn.Linear, self.hidden, **self.kw)(action)
    action_proj = nn.act(self.act)(self.sub('action_proj_norm', nn.Norm, self.norm)(action_proj))

    # 3. Context projections (separate, gradient-isolated from fine pathway)
    stoch_flat_sg = sg(stoch_flat)  # prevent coarse losses from affecting posterior
    stoch_proj_ctx = self.sub('stoch_proj_ctx', nn.Linear, self.coarse_hidden, **self.kw)(stoch_flat_sg)
    stoch_proj_ctx = nn.act(self.act)(self.sub('stoch_proj_ctx_norm', nn.Norm, self.norm)(stoch_proj_ctx))
    action_proj_ctx = self.sub('action_proj_ctx', nn.Linear, self.coarse_hidden, **self.kw)(action)
    action_proj_ctx = nn.act(self.act)(self.sub('action_proj_ctx_norm', nn.Norm, self.norm)(action_proj_ctx))

    # 4. Context BlockGRU + boundary gate
    ctx_before_gate = ctx
    ctx_after_gru = self._context_core(ctx, stoch_proj_ctx, action_proj_ctx)
    gate_prob, gate_binary, gate = self._boundary_gate(stoch_proj_ctx, action_proj_ctx, ctx)
    # Force context update at t=1 of episode (first step with real z, a)
    force_gate = prev_reset > 0.5
    gate = jnp.where(force_gate[:, None], jnp.ones_like(gate), gate)
    gate_binary = jnp.where(force_gate, jnp.ones_like(gate_binary), gate_binary)
    ctx = nn.cast(gate * ctx_after_gru + (1 - gate) * ctx)

    # 5. GRU core (fine pathway — no context input)
    deter = self._core(deter, stoch_proj, action_proj)

    # 6. Posterior from [deter, tokens] (no context)
    tokens = tokens.reshape((*deter.shape[:-1], -1))
    if self.absolute:
      x = tokens
    else:
      x = jnp.concatenate([deter, tokens], -1)
    for i in range(self.obslayers):
      x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
    logit = self._logit('obslogit', x)
    stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))

    # 7. Coarse prior from [context, sg(z), action]
    coarse_logit = self._coarse_prior(ctx, stoch_flat_sg, action)

    carry = dict(deter=deter, stoch=stoch, context=ctx,
                prev_reset=nn.cast(f32(reset)))
    feat = dict(deter=deter, stoch=stoch, logit=logit, context=ctx,
                coarse_logit=coarse_logit, gate_prob=gate_prob,
                gate_binary=gate_binary,
                ctx_before_gate=ctx_before_gate,
                ctx_after_gru=nn.cast(ctx_after_gru))
    entry = dict(deter=deter, stoch=stoch, context=ctx,
                 prev_reset=nn.cast(f32(reset)))
    assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
    return carry, (entry, feat)

  def imagine(self, carry, policy, length, training, single=False):
    if single:
      action = policy(sg(carry)) if callable(policy) else policy
      actemb = nn.DictConcat(self.act_space, 1)(action)
      actemb /= sg(jnp.maximum(1, jnp.abs(actemb)))

      # Fine projections
      stoch_flat = carry['stoch'].reshape((carry['stoch'].shape[0], -1))
      stoch_proj = self.sub('stoch_proj', nn.Linear, self.hidden, **self.kw)(stoch_flat)
      stoch_proj = nn.act(self.act)(self.sub('stoch_proj_norm', nn.Norm, self.norm)(stoch_proj))
      action_proj = self.sub('action_proj', nn.Linear, self.hidden, **self.kw)(actemb)
      action_proj = nn.act(self.act)(self.sub('action_proj_norm', nn.Norm, self.norm)(action_proj))

      # Context projections (separate)
      stoch_flat_sg = sg(stoch_flat)
      stoch_proj_ctx = self.sub('stoch_proj_ctx', nn.Linear, self.coarse_hidden, **self.kw)(stoch_flat_sg)
      stoch_proj_ctx = nn.act(self.act)(self.sub('stoch_proj_ctx_norm', nn.Norm, self.norm)(stoch_proj_ctx))
      action_proj_ctx = self.sub('action_proj_ctx', nn.Linear, self.coarse_hidden, **self.kw)(actemb)
      action_proj_ctx = nn.act(self.act)(self.sub('action_proj_ctx_norm', nn.Norm, self.norm)(action_proj_ctx))

      # Context BlockGRU + boundary gate
      ctx = carry['context']
      ctx_new = self._context_core(ctx, stoch_proj_ctx, action_proj_ctx)
      gate_prob, gate_binary, gate = self._boundary_gate(stoch_proj_ctx, action_proj_ctx, ctx)
      ctx = nn.cast(gate * ctx_new + (1 - gate) * ctx)

      # GRU core (no context)
      deter = self._core(carry['deter'], stoch_proj, action_proj)

      # Fine prior (no context)
      logit = self._prior(deter)
      stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))

      # Coarse prior from [context, sg(z), action]
      coarse_logit = self._coarse_prior(ctx, sg(stoch_flat), actemb)

      carry = nn.cast(dict(deter=deter, stoch=stoch, context=ctx,
                          prev_reset=jnp.zeros(deter.shape[0], f32)))
      feat = nn.cast(dict(deter=deter, stoch=stoch, logit=logit, context=ctx,
                          coarse_logit=coarse_logit, gate_prob=gate_prob,
                          gate_binary=gate_binary))
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

    # Fine prior KL
    prior = self._prior(feat['deter'])
    post = feat['logit']
    dyn_raw = self._dist(sg(post)).kl(self._dist(prior))
    rep_raw = self._dist(post).kl(self._dist(sg(prior)))
    dyn = jnp.maximum(dyn_raw, self.free_nats) if self.free_nats else dyn_raw
    rep = jnp.maximum(rep_raw, self.free_nats) if self.free_nats else rep_raw

    # Coarse prior KL (trains coarse prior toward posterior, no reverse)
    coarse_prior = feat['coarse_logit']
    coarse_dyn = self._dist(sg(post)).kl(self._dist(coarse_prior))
    if self.free_nats:
      coarse_dyn = jnp.maximum(coarse_dyn, self.free_nats)

    # Boundary gate sparsity: sequence-level rate penalty
    # Multiply by T so per-timestep gradient is f'(rate), not f'(rate)/T
    gate_prob = feat['gate_prob']  # [B, T]
    T_ = gate_prob.shape[-1]
    gate_rate = gate_prob.mean(-1, keepdims=True)  # [B, 1]
    sparse = (jax.nn.relu(gate_rate - self.boundary_prior) ** 2
              + jax.nn.relu(self.min_gate_rate - gate_rate) ** 2) * T_
    sparse = jnp.broadcast_to(sparse, gate_prob.shape)

    # Gate-info: incentivize gate to fire after surprising observations.
    # Lagged KL surprise: high KL at t-1 → gate should fire at t to store
    # the new information in context. sg() prevents gaming the KL.
    surprise = sg(dyn_raw)  # [B, T]
    surprise_prev = jnp.concatenate(
        [jnp.zeros_like(surprise[:, :1]), surprise[:, :-1]], axis=1)
    surprise_norm = surprise_prev / (surprise_prev.mean(-1, keepdims=True) + 1e-8)
    gate_info = -gate_prob * surprise_norm

    losses = {
        'dyn': dyn, 'rep': rep,
        'coarse_dyn': coarse_dyn,
        'sparse': sparse,
        'gate_info': gate_info,
    }
    metrics['dyn_ent'] = self._dist(prior).entropy().mean()
    metrics['rep_ent'] = self._dist(post).entropy().mean()
    metrics['coarse_ent'] = self._dist(coarse_prior).entropy().mean()
    metrics['gate_prob_mean'] = gate_prob.mean()
    metrics['gate_prob_std'] = gate_prob.std(-1).mean()
    metrics['gate_prob_t1'] = gate_prob[:, 1].mean()
    metrics['gate_info_surprise'] = surprise_prev.mean()
    metrics['surprise_mean'] = surprise.mean()
    metrics['surprise_std'] = surprise.std(-1).mean()
    metrics['surprise_t0'] = surprise[:, 0].mean()
    metrics['surprise_t1'] = surprise[:, 1].mean()
    metrics['surprise_norm_t1'] = surprise_norm[:, 1].mean()
    if not self.stochastic_gate:
      metrics['gate_change_freq'] = f32(gate_prob > 0.5).mean()
    feat['surprise'] = dyn_raw
    return carry, entries, losses, feat, metrics

  def _context_core(self, context, stoch_proj_ctx, action_proj_ctx):
    """BlockGRU on context dim, using same number of blocks as fine GRU."""
    g = self.blocks
    flat2group = lambda x: einops.rearrange(x, '... (g h) -> ... g h', g=g)
    group2flat = lambda x: einops.rearrange(x, '... g h -> ... (g h)', g=g)

    # Project context hidden state
    x0 = self.sub('ctxin0', nn.Linear, self.coarse_hidden, **self.kw)(context)
    x0 = nn.act(self.act)(self.sub('ctxin0norm', nn.Norm, self.norm)(x0))

    parts = [x0, stoch_proj_ctx, action_proj_ctx]
    x = jnp.concatenate(parts, -1)[..., None, :].repeat(g, -2)
    x = group2flat(jnp.concatenate([flat2group(context), x], -1))
    for i in range(self.dynlayers):
      x = self.sub(f'ctxhid{i}', nn.BlockLinear, self.context, g, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'ctxhid{i}norm', nn.Norm, self.norm)(x))
    x = self.sub('ctxgru', nn.BlockLinear, 3 * self.context, g, **self.kw)(x)
    gates = jnp.split(flat2group(x), 3, -1)
    reset, cand, update = [group2flat(x) for x in gates]
    reset = jax.nn.sigmoid(reset)
    cand = jnp.tanh(reset * cand)
    update = jax.nn.sigmoid(update - 1)
    context_new = update * cand + (1 - update) * context
    return context_new

  def _boundary_gate(self, stoch_proj_ctx, action_proj_ctx, context):
    """Scalar binary boundary gate with Bernoulli sampling + STE.

    Returns (gate_prob, gate_sample) where gate_sample is binary {0, 1}.
    """
    x = jnp.concatenate([stoch_proj_ctx, action_proj_ctx, context], -1)
    gate_logit = self.sub('boundary_gate', nn.Linear, 1, **self.kw)(x)
    gate_logit = gate_logit.squeeze(-1)  # [B]
    prob = jax.nn.sigmoid(gate_logit)
    # Binary gate via straight-through estimator
    if self.stochastic_gate:
      binary = jax.random.bernoulli(nj.seed(), prob).astype(f32)
    else:
      binary = f32(prob > 0.5)
    gate = sg(binary - prob) + prob
    # Expand to broadcast with context dim
    gate = gate[..., None]  # [B, 1]
    return prob, binary, gate

  def _core(self, deter, stoch_proj, action_proj):
    """GRU core with 3 inputs: deter, stoch_proj, action_proj. No context."""
    g = self.blocks
    flat2group = lambda x: einops.rearrange(x, '... (g h) -> ... g h', g=g)
    group2flat = lambda x: einops.rearrange(x, '... g h -> ... (g h)', g=g)

    x0 = self.sub('dynin0', nn.Linear, self.hidden, **self.kw)(deter)
    x0 = nn.act(self.act)(self.sub('dynin0norm', nn.Norm, self.norm)(x0))
    parts = [x0, stoch_proj, action_proj]

    x = jnp.concatenate(parts, -1)[..., None, :].repeat(g, -2)
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

  def _prior(self, deter):
    """Fine prior: MLP from deter only (no context)."""
    x = deter
    for i in range(self.imglayers):
      x = self.sub(f'prior{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'prior{i}norm', nn.Norm, self.norm)(x))
    return self._logit('priorlogit', x)

  def context_step(self, context, stoch_flat, action_emb):
    """Run one coarse GRU update without boundary gate (for HL rollouts).

    Reuses the same submodules as _observe/_imagine context pathway.
    No sg() needed on inputs — entire vlong is sg'd downstream.
    """
    stoch_proj_ctx = self.sub('stoch_proj_ctx', nn.Linear, self.coarse_hidden, **self.kw)(stoch_flat)
    stoch_proj_ctx = nn.act(self.act)(self.sub('stoch_proj_ctx_norm', nn.Norm, self.norm)(stoch_proj_ctx))
    action_proj_ctx = self.sub('action_proj_ctx', nn.Linear, self.coarse_hidden, **self.kw)(action_emb)
    action_proj_ctx = nn.act(self.act)(self.sub('action_proj_ctx_norm', nn.Norm, self.norm)(action_proj_ctx))
    return self._context_core(context, stoch_proj_ctx, action_proj_ctx)

  def _coarse_prior(self, context, stoch_flat, action_emb):
    """Coarse prior: MLP from [context, z, action]."""
    x = jnp.concatenate([context, stoch_flat, action_emb], -1)
    for i in range(self.imglayers):
      x = self.sub(f'coarse{i}', nn.Linear, self.coarse_hidden, **self.kw)(x)
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


def _bernoulli_kl(p, q):
  """KL(Bernoulli(p) || Bernoulli(q)), element-wise.

  Args:
    p: predicted probability (any shape)
    q: prior probability (scalar or broadcastable)

  Returns:
    KL divergence, same shape as p
  """
  eps = 1e-6
  p = jnp.clip(p, eps, 1 - eps)
  q = jnp.clip(jnp.broadcast_to(jnp.asarray(q, dtype=p.dtype), p.shape), eps, 1 - eps)
  return p * jnp.log(p / q) + (1 - p) * jnp.log((1 - p) / (1 - q))
