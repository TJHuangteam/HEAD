
import os

from metadrive.envs.scenario_env import ScenarioEnv



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import logging

from HEAD.algorithm.SAC.SAC_learner import SAC_Learner, SACConfig
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from functools import partial
from metadrive.scenario.utils import get_number_of_scenarios

desc = "Load a database to simulator and replay scenarios"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument("--database_path", "-d", default="/path/to/your/database", help="The path of the database")
parser.add_argument("--render", default="none", choices=["none", "2D", "3D", "advanced", "semantic"])
parser.add_argument("--scenario_index", default=None, type=int, help="Specifying a scenario to run")
args = parser.parse_args()

database_path = os.path.abspath(args.database_path)
num_scenario = get_number_of_scenarios(database_path)

def create_env(need_monitor=False):

    env = ScenarioEnv(
        {
            "use_render": True,   ## args.render == "3D" or args.render == "advanced"
            # "agent_policy": EnvInputPolicy,
            "manual_control": False,
            "render_pipeline": False,
            "show_interface": True,
            "show_logo": False,
            "show_fps": False,
            "log_level": logging.CRITICAL,
            "num_scenarios": 1440,
            "interface_panel": [],
            "horizon": 1000,
            "vehicle_config": dict(
                show_navi_mark=True,
                show_line_to_dest=True,
                show_dest_mark=True,
                no_wheel_friction=False,
                lidar=dict(num_lasers=120, distance=50, num_others=4),
                lane_line_detector=dict(num_lasers=12, distance=50),
                side_detector=dict(num_lasers=160, distance=50)
            ),
            "data_directory": database_path,
        }
    )

    if need_monitor:
        env = Monitor(env)
    return env


def parse_args_cfgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_flag', type=int, default=0,
                        help='train = 1 or eval = 0')
    parser.add_argument('--train_name', type=str, default='15e5_1025_cut1')
    parser.add_argument('--total_steps', type=float, default=2e6) # 1e6
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args_cfgs()
    SAC_cfg = SACConfig(args)


    if bool(args.train_flag):
        env = create_env()
        # env = create_env(True)
        # env.reset()
        # env.agent.expert_takeover = True
        env_num = 1
        train_envs = SubprocVecEnv([partial(create_env, True) for _ in range(env_num)])
        SAC = SAC_Learner(train_envs, SAC_cfg)
        # env = create_env(True)
        # SAC = SAC_Learner(env, SAC_cfg)

        SAC.agent_initialize()
        SAC.train()
        SAC.save()

    else:
        eval_env = create_env()

        SAC = SAC_Learner(eval_env, SAC_cfg)
        SAC.agent_initialize()
        SAC.load()
        SAC.eval()
