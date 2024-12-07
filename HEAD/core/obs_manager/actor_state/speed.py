"""Adapted from https://github.com/zhejz/carla-roach/ CC-BY-NC 4.0 license."""

import numpy as np
from gym import spaces

from HEAD.core.obs_manager.obs_manager import ObsManagerBase


class ObsManager(ObsManagerBase):
    """
    in m/s
    """

    def __init__(self, obs_configs):
        self._parent_actor = None
        super(ObsManager, self).__init__()

    def _define_obs_space(self):
        self.obs_space = spaces.Dict({
            'speed': spaces.Box(low=-10.0, high=30.0, shape=(1,), dtype=np.float32),
            'speed_xy': spaces.Box(low=-10.0, high=30.0, shape=(1,), dtype=np.float32),
            'forward_speed': spaces.Box(low=-10.0, high=30.0, shape=(1,), dtype=np.float32)
        })

    def attach_ego_vehicle(self, parent_actor):
        self._parent_actor = parent_actor

    def get_observation(self, vehicle, env):

        speed = vehicle.speed
        speed_xy = speed
        forward_speed = speed

        obs = {
            'speed': np.array([speed], dtype=np.float32),
            'speed_xy': np.array([speed_xy], dtype=np.float32),
            'forward_speed': np.array([forward_speed], dtype=np.float32)
        }
        return obs

    def clean(self):
        self._parent_actor = None
