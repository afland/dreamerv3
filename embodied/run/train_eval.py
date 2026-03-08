import collections
from functools import partial as bind

import elements
import embodied
import numpy as np

try:
  import imageio.v2 as imageio
  HAS_IMAGEIO = True
except ImportError:
  HAS_IMAGEIO = False


def _render_gate_overlay(frame, gate_prob, bar_height=6):
  """Render frame with a gate probability bar at the bottom.

  A horizontal bar shows gate_prob as a fill level (0=empty, 1=full).
  Bar color: green-to-red gradient based on probability value.
  Background of bar is dark gray so empty regions are visible.
  """
  H, W, C = frame.shape
  out = np.copy(frame)
  p = float(np.clip(gate_prob, 0, 1))
  fill_w = int(p * W)
  # Color: green at 0, yellow at 0.5, red at 1
  if p < 0.5:
    r = int(255 * (p / 0.5))
    g = 255
  else:
    r = 255
    g = int(255 * ((1 - p) / 0.5))
  bar_color = np.array([r, g, 0], dtype=np.uint8)[:C]
  bg_color = np.array([40, 40, 40], dtype=np.uint8)[:C]
  # Draw bar background
  out[-bar_height:, :] = bg_color
  # Draw filled portion
  if fill_w > 0:
    out[-bar_height:, :fill_w] = bar_color
  return out


def _save_eval_video(result, videodir, step_num):
  """Save eval episode video with optional context visualization."""
  # Find the first policy_* image key
  img_key = None
  for k in result:
    if k.startswith('policy_') and isinstance(result[k], np.ndarray):
      if result[k].ndim == 4 and result[k].dtype == np.uint8:
        img_key = k
        break
  if img_key is None:
    return
  frames = result.pop(img_key)  # (T, H, W, C)
  gate_probs = result.pop('gate_prob', None)  # (T,) or None
  result.pop('context', None)  # remove from result to avoid logging issues
  if gate_probs is not None:
    rendered = []
    for t in range(len(frames)):
      rendered.append(_render_gate_overlay(frames[t], gate_probs[t]))
    frames = np.stack(rendered)
  videodir.mkdir()
  path = str(videodir / f'eval_{step_num}.mp4')
  try:
    imageio.mimwrite(path, frames, fps=15, codec='libx264')
  except Exception:
    imageio.mimwrite(path, frames, fps=15, codec='mpeg4')
  print(f'Saved eval video: {path}')


def train_eval(
    make_agent,
    make_replay_train,
    make_replay_eval,
    make_env_train,
    make_env_eval,
    make_stream,
    make_logger,
    args):

  agent = make_agent()
  replay_train = make_replay_train()
  replay_eval = make_replay_eval()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  logdir.mkdir()
  videodir = logdir / 'videos'
  print('Logdir', logdir)
  step = logger.step
  usage = elements.Usage(**args.usage)
  agg = elements.Agg()
  train_episodes = collections.defaultdict(elements.Agg)
  train_epstats = elements.Agg()
  eval_episodes = collections.defaultdict(elements.Agg)
  eval_epstats = elements.Agg()
  policy_fps = elements.FPS()
  train_fps = elements.FPS()

  batch_steps = args.batch_size * args.batch_length
  should_train = elements.when.Ratio(args.train_ratio / batch_steps)
  should_log = elements.when.Clock(args.log_every)
  should_report = elements.when.Clock(args.report_every)
  should_save = elements.when.Clock(args.save_every)

  @elements.timer.section('logfn')
  def logfn(tran, worker, mode):
    episodes = dict(train=train_episodes, eval=eval_episodes)[mode]
    epstats = dict(train=train_epstats, eval=eval_epstats)[mode]
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')
    for key, value in tran.items():
      if value.dtype == np.uint8 and value.ndim == 3:
        if worker == 0:
          episode.add(f'policy_{key}', value, agg='stack')
      elif key.startswith('log/'):
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key + '/avg', value, agg='avg')
        episode.add(key + '/max', value, agg='max')
        episode.add(key + '/sum', value, agg='sum')
    if mode == 'eval' and worker == 0:
      if 'gate_prob' in tran:
        episode.add('gate_prob', np.array(tran['gate_prob']), agg='stack')
    if tran['is_last']:
      result = episode.result()
      logger.add({
          'score': result.pop('score'),
          'length': result.pop('length'),
      }, prefix='episode')
      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      if mode == 'eval' and worker == 0 and HAS_IMAGEIO:
        _save_eval_video(result, videodir, int(step))
      epstats.add(result)

  fns = [bind(make_env_train, i) for i in range(args.envs)]
  driver_train = embodied.Driver(fns, parallel=(not args.debug))
  driver_train.on_step(lambda tran, _: step.increment())
  driver_train.on_step(lambda tran, _: policy_fps.step())
  _REPLAY_EXCLUDE = ('context', 'gates', 'gate_prob')
  driver_train.on_step(lambda tran, _: replay_train.add(
      {k: v for k, v in tran.items() if k not in _REPLAY_EXCLUDE}, _))
  driver_train.on_step(bind(logfn, mode='train'))

  fns = [bind(make_env_eval, i) for i in range(args.eval_envs)]
  driver_eval = embodied.Driver(fns, parallel=(not args.debug))
  driver_eval.on_step(lambda tran, _: replay_eval.add(
      {k: v for k, v in tran.items() if k not in _REPLAY_EXCLUDE}, _))
  driver_eval.on_step(bind(logfn, mode='eval'))
  driver_eval.on_step(lambda tran, _: policy_fps.step())

  stream_train = iter(agent.stream(make_stream(replay_train, 'train')))
  stream_report = iter(agent.stream(make_stream(replay_train, 'report')))
  stream_eval = iter(agent.stream(make_stream(replay_eval, 'eval')))

  carry_train = [agent.init_train(args.batch_size)]
  carry_report = agent.init_report(args.batch_size)
  carry_eval = agent.init_report(args.batch_size)

  def trainfn(tran, worker):
    if len(replay_train) < args.batch_size * args.batch_length:
      return
    for _ in range(should_train(step)):
      with elements.timer.section('stream_next'):
        batch = next(stream_train)
      carry_train[0], outs, mets = agent.train(carry_train[0], batch)
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay_train.update(outs['replay'])
      agg.add(mets, prefix='train')
  driver_train.on_step(trainfn)

  def reportfn(carry, stream):
    agg = elements.Agg()
    for _ in range(args.report_batches):
      batch = next(stream)
      carry, mets = agent.report(carry, batch)
      agg.add(mets)
    return carry, agg.result()

  cp = elements.Checkpoint(logdir / 'ckpt')
  cp.step = step
  cp.agent = agent
  cp.replay_train = replay_train
  cp.replay_eval = replay_eval
  if args.from_checkpoint:
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=bind(agent.load, regex=args.from_checkpoint_regex)))
  cp.load_or_save()
  should_save(step)  # Register that we just saved.

  print('Start training loop')
  train_policy = lambda *args: agent.policy(*args, mode='train')
  eval_policy = lambda *args: agent.policy(*args, mode='eval')
  driver_train.reset(agent.init_policy)
  while step < args.steps:

    if should_report(step):
      print('Evaluation')
      driver_eval.reset(agent.init_policy)
      driver_eval(eval_policy, episodes=args.eval_eps)
      logger.add(eval_epstats.result(), prefix='epstats')
      if len(replay_train):
        carry_report, mets = reportfn(carry_report, stream_report)
        logger.add(mets, prefix='report')
      if len(replay_eval):
        carry_eval, mets = reportfn(carry_eval, stream_eval)
        logger.add(mets, prefix='eval')

    driver_train(train_policy, steps=10)

    if should_log(step):
      logger.add(agg.result())
      logger.add(train_epstats.result(), prefix='epstats')
      logger.add(replay_train.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()

    if should_save(step):
      cp.save()

  logger.close()
