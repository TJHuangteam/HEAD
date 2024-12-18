from metadrive.envs import MetaDriveEnv
from metadrive.policy.lange_change_policy import LaneChangePolicy
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from metadrive.component.map.base_map import BaseMap
from metadrive.utils import generate_gif
from IPython.display import Image


def create_env(need_monitor=False):
    env = MetaDriveEnv(dict(map="SSSSSSSSSSS",
                            # This policy setting simplifies the task
                            discrete_action=True,
                            discrete_throttle_dim=3,
                            discrete_steering_dim=3,
                            horizon=500,
                            # scenario setting
                            random_spawn_lane_index=False,
                            num_scenarios=1,
                            start_seed=5,
                            traffic_density=0,
                            accident_prob=0,
                            log_level=50))
    if need_monitor:
        env = Monitor(env)
    return env


env = create_env()
env.reset()
ret = env.render(mode="topdown",
                 window=False,
                 screen_size=(600, 600),
                 camera_position=(50, 50))
env.close()
plt.axis("off")
plt.imshow(ret)
plt.show()
