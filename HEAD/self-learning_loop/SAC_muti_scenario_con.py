
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse

from HEAD.algorithm.SAC.SAC_learner import SAC_Learner, SACConfig
from stable_baselines3.common.monitor import Monitor
from metadrive.envs import MetaDriveEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from functools import partial


def create_env(need_monitor=False):
    # map = "SSSSSS"
    # map = "XCO"
    env = MetaDriveEnv(dict(map="O",  # XCO
                            # This policy setting simplifives the task
                            discrete_action=False,
                            horizon=1000,
                            use_render=False,
                            random_traffic=True,
                            # scenario setting
                            random_spawn_lane_index=False,
                            num_scenarios=1,
                            start_seed=5,
                            traffic_density=0.2,
                            accident_prob=0,
                            use_lateral_reward=True,
                            log_level=50))

    if need_monitor:
        env = Monitor(env)
    return env


def parse_args_cfgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_flag', type=int, default=0,
                        help='train = 1 or eval = 0')
    parser.add_argument('--train_name', type=str, default='O')
    parser.add_argument('--total_steps', type=float, default=1e6)  # 1e6
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args_cfgs()
    SAC_cfg = SACConfig(args)


    if bool(args.train_flag):
        env = create_env()
        env_num = 1
        train_envs = SubprocVecEnv([partial(create_env, True) for _ in range(env_num)])
        SAC = SAC_Learner(train_envs, SAC_cfg)

        SAC.agent_initialize()
        SAC.train()
        SAC.save()

    else:
        eval_env = create_env()

        SAC = SAC_Learner(eval_env, SAC_cfg)
        SAC.agent_initialize()
        SAC.load()
        SAC.eval()
