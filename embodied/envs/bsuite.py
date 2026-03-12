import embodied
import numpy as np


class BSuite(embodied.Env):

  def __init__(self, task):
    np.int = int  # Patch deprecated Numpy alias used inside BSuite.
    from . import from_dm
    if '/' not in task:
      task = f'{task}/0'
    import bsuite
    env = bsuite.load_from_id(task)
    self.num_episodes = 0
    env = from_dm.FromDM(env)
    self.env = env

  @property
  def obs_space(self):
    return self.env.obs_space

  @property
  def act_space(self):
    return self.env.act_space

  def step(self, action):
    obs = self.env.step(action)
    if obs['is_last']:
      self.num_episodes += 1
    return obs
