import copy
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from typing import Union, Tuple, Dict

import yaml
import numpy as np
from metadrive.component.sensors.lidar import Lidar
from metadrive.component.algorithm.blocks_prob_dist import PGBlockDistConfig
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import parse_map_config, MapGenerateMethod
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.constants import DEFAULT_AGENT, TerminationState
from metadrive.manager.traffic_manager import TrafficMode
from metadrive.utils import Config
from HEAD.core.obs_manager.obs_collect import EnvironmentInfoCollector
from HEAD.envs.base_env import BaseEnv
from config import cfg, cfg_from_yaml_file


SACENV_DEFAULT_CONFIG = dict(
    # ===== Generalization =====
    start_seed=0,
    num_scenarios=1,  # single scenario
    use_render=True,

    # ===== PG Map Config =====
    map = "SSSSSSSS" ,  # int or string: an easy way to fill map_config, straight
    block_dist_config=PGBlockDistConfig,
    random_lane_width=False,
    random_lane_num=False,
    map_config={
        BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
        BaseMap.GENERATE_CONFIG: None,  # it can be a file path / block num / block ID sequence
        BaseMap.LANE_WIDTH: 3.5,
        BaseMap.LANE_NUM: 4,
        "exit_length": 50,
    },
    store_map=True,

    # ===== Traffic =====
    traffic_density=0.0,
    need_inverse_traffic=False,
    traffic_mode=TrafficMode.Trigger,  # "Respawn", "Trigger"
    random_traffic=False,  # Traffic is randomized at default.
    # this will update the vehicle_config and set to traffic
    traffic_vehicle_config=dict(
        show_navi_mark=False,
        show_dest_mark=False,
        enable_reverse=False,
        show_lidar=True,
        show_lane_line_detector=False,
        show_side_detector=False,
    ),

    # ===== Object =====
    accident_prob=0.,  # accident may happen on each block with this probability, except multi-exits block
    static_traffic_object=True,  # object won't react to any collisions

    # ===== Others =====
    use_AI_protector=False,
    save_level=0.5,

    # ===== Agent =====
    random_spawn_lane_index=True,
    sensors=dict(lidar=(Lidar, )),
    vehicle_config=dict(
            navigation_module = NodeNetworkNavigation,
            agent_policy= None,
            lidar=dict(num_lasers=240, distance=40, num_others=0)
                        ),
    agent_configs={
        DEFAULT_AGENT: dict(
            use_special_color=True,
            spawn_lane_index=(FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0),
            lidar=dict(num_lasers=240, distance=40, num_others=0)
        )
    },

    # ===== Termination Scheme =====
    out_of_route_done=False,
    on_continuous_line_done=True,
    crash_vehicle_done=True,
    crash_object_done=True,
    crash_human_done=True,

    # ===== Multi-agent =====
    is_multi_agent=False,

    # ===== RL-Settings =====
    horizon=1000
)


# ==============================================================================
# -- Util -----------------------------------------------------------
# ==============================================================================
def lamp(v, x, y):
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0] + 1e-10)

def parse_args_cfgs():
    cfg_file_path = "./config.yaml"
    cfg_from_yaml_file(cfg_file_path, cfg)
    cfg.EXP_GROUP_PATH = '/'.join(cfg_file_path.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    return cfg

def read_yaml_file(filepath):
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)
    return data


env_config_dict = read_yaml_file('./HEAD/config/env_config.yaml')  # Read the YAML file

class SACEnv(BaseEnv):
    @classmethod ## how to config,
    def default_config(cls) -> Config:
        config = super(SACEnv, cls).default_config()
        # config.update(env_config_dict)
        config.update(SACENV_DEFAULT_CONFIG)
        config.register_type("map", str, int)
        config["map_config"].register_type("config", None)
        return config

    def __init__(self):
        self.default_config_copy = Config(self.default_config(), unchangeable=True)
        super(SACEnv, self).__init__()

        # scenario setting
        self.start_seed = self.start_index = self.config["start_seed"]
        self.env_num = self.num_scenarios = self.config["num_scenarios"]

        # constraints
        cfg = parse_args_cfgs()
        self.targetSpeed = float(cfg.GYM_ENV.TARGET_SPEED)
        self.maxSpeed = float(cfg.GYM_ENV.MAX_SPEED)
        self.minSpeed = float(cfg.GYM_ENV.MIN_SPEED)
        self.LANE_WIDTH = float(cfg.CARLA.LANE_WIDTH)
        self.N_SPAWN_CARS = int(cfg.TRAFFIC_MANAGER.N_SPAWN_CARS)

    def _post_process_config(self, config):
        config = super(SACEnv, self)._post_process_config(config)
        if not config["norm_pixel"]:
            self.logger.warning(
                "You have set norm_pixel = False, which means the observation will be uint8 values in [0, 255]. "
                "Please make sure you have parsed them later before feeding them to network!"
            )

        config["map_config"] = parse_map_config(
            easy_map_config=config["map"], new_map_config=config["map_config"], default_config=self.default_config_copy
        )

        config["vehicle_config"]["norm_pixel"] = config["norm_pixel"]
        config["vehicle_config"]["random_agent_model"] = config["random_agent_model"]
        target_v_config = copy.deepcopy(config["vehicle_config"])
        if not config["is_multi_agent"]:
            target_v_config.update(config["agent_configs"][DEFAULT_AGENT])
            config["agent_configs"][DEFAULT_AGENT] = target_v_config
        return config

    def done_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        self.done = False
        max_step = self.config["horizon"] is not None and self.episode_lengths[vehicle_id] >= self.config["horizon"]
        done_info = {
            TerminationState.CRASH_VEHICLE: vehicle.crash_vehicle,
            TerminationState.CRASH_OBJECT: vehicle.crash_object,
            TerminationState.CRASH_BUILDING: vehicle.crash_building,
            TerminationState.CRASH_HUMAN: vehicle.crash_human,
            TerminationState.CRASH_SIDEWALK: vehicle.crash_sidewalk,
            TerminationState.OUT_OF_ROAD: self._is_out_of_road(vehicle),
            TerminationState.SUCCESS: self._is_arrive_destination(vehicle),
            TerminationState.MAX_STEP: max_step,
            TerminationState.ENV_SEED: self.current_seed,
            # TerminationState.CURRENT_BLOCK: self.agent.navigation.current_road.block_ID(),
            # crash_vehicle=False, crash_object=False, crash_building=False, out_of_road=False, arrive_dest=False,
        }

        # for compatibility
        # crash almost equals to crashing with vehicles
        done_info[TerminationState.CRASH] = (
            done_info[TerminationState.CRASH_VEHICLE] or done_info[TerminationState.CRASH_OBJECT]
            or done_info[TerminationState.CRASH_BUILDING] or done_info[TerminationState.CRASH_SIDEWALK]
            or done_info[TerminationState.CRASH_HUMAN]
        )

        # determine env return
        if done_info[TerminationState.SUCCESS]:
            self.done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: arrive_dest.".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.OUT_OF_ROAD]:
            self.done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: out_of_road.".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_VEHICLE] and self.config["crash_vehicle_done"]:
            self.done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: crash vehicle ".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_OBJECT] and self.config["crash_object_done"]:
            self.done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: crash object ".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_BUILDING]:
            self.done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: crash building ".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_HUMAN] and self.config["crash_human_done"]:
            self.done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: crash human".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.MAX_STEP]:
            # single agent horizon has the same meaning as max_step_per_agent
            if self.config["truncate_as_terminate"]:
                self.done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: max step ".format(self.current_seed),
                extra={"log_once": True}
            )
        return self.done, done_info
    def cost_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        step_info = dict()
        step_info["cost"] = 0
        return step_info['cost'], step_info

    @staticmethod
    def _is_arrive_destination(vehicle):
        """
        Args:
            vehicle: The BaseVehicle instance.

        Returns:
            flag: Whether this vehicle arrives its destination.
        """
        long, lat = vehicle.navigation.final_lane.local_coordinates(vehicle.position)
        flag = (vehicle.navigation.final_lane.length - 5 < long < vehicle.navigation.final_lane.length + 5) and (
                vehicle.navigation.get_current_lane_width() / 2 >= lat >=
                (0.5 - vehicle.navigation.get_current_lane_num()) * vehicle.navigation.get_current_lane_width()
        )
        return flag

    def _is_out_of_road(self, vehicle):
        # A specified function to determine whether this vehicle should be done.
        # return vehicle.on_yellow_continuous_line or (not vehicle.on_lane) or vehicle.crash_sidewalk
        ret = not vehicle.on_lane
        if self.config["out_of_route_done"]:
            ret = ret or vehicle.out_of_route
        elif self.config["on_continuous_line_done"]:
            ret = ret or vehicle.on_yellow_continuous_line or vehicle.on_white_continuous_line or vehicle.crash_sidewalk
        return ret


    def reward_function(self, vehicle_id: str):
        """
        Override this func to get a new reward function
        :param vehicle_id: id of BaseVehicle
        :return: reward
        """
        vehicle = self.agents[vehicle_id]
        step_info = dict()
        lidar = self.engine.get_sensor("lidar")
        '''******   Reward Design   ******'''
        # 势场奖励
        # s_len = 5.037  ##车长
        # d_width = 2.077  ##车宽
        # k_f = 0.001
        # deta_1 = 8
        # deta_2 = 10
        # reward_factor = -18
        # scale = math.exp(((0.8 - d_width) ** 2 / deta_1 ** 2) + ((20 - s_len) ** 2 / deta_2 ** 2))  ###
        # k = -reward_factor / (k_f * scale - k_f)
        # b = reward_factor - k * k_f
        # reward_dis = 0
        # # dd = abs(state_vector[1]) ###
        # dd = 5 ###
        # # ds = abs(state_vector[0]) ###
        # ds = 3.5 ###
        # r = min((math.exp((min(max(dd - d_width, 0.0), 150) ** 2 / deta_1 ** 2) + (
        #         (min(max(ds - s_len, 0.0), 150)) ** 2 / deta_2 ** 2)) * k_f * k + b), 0)
        # reward_dis += r * 0.15
        # # print('dis_reward:',reward_dis)

        v_S = vehicle.speed

        # 速度奖励
        scaled_speed_l = lamp(v_S, [0, self.minSpeed], [0, 1])
        scaled_speed_h = lamp(v_S, [self.minSpeed, self.maxSpeed], [0, 1])
        reward_hs_l = 0.5
        reward_hs_h = 4.0
        reward_speed = reward_hs_l * np.clip(scaled_speed_l, 0, 1) + reward_hs_h * np.clip(scaled_speed_h, 0, 1)

        # 碰撞惩罚
        collision = vehicle.crash_vehicle or vehicle.crash_sidewalk or vehicle.crash_human or vehicle.crash_human or vehicle.crash_building or self._is_out_of_road(vehicle)
        if collision:
            reward_cl = -30.0
        else:
            reward_cl = 0.0

        reward = reward_cl + reward_speed
        step_info["step_reward"] = reward

        return reward, step_info

    def obs_function(self, vehicle_id: str) :

        config_path = "./HEAD/config/obs_test.yaml"
        config_dict = read_yaml_file(config_path)
        vehicle = self.agents[vehicle_id]
        obs_collector = EnvironmentInfoCollector(self, vehicle, config_dict)
        obs = obs_collector.get_observation()

        return obs, obs

    def setup_engine(self):
        super(SACEnv, self).setup_engine()
        from metadrive.manager.traffic_manager import PGTrafficManager
        from metadrive.manager.pg_map_manager import PGMapManager
        from metadrive.manager.object_manager import TrafficObjectManager
        self.engine.register_manager("map_manager", PGMapManager())
        self.engine.register_manager("traffic_manager", PGTrafficManager())
        if abs(self.config["accident_prob"] - 0) > 1e-2:
            self.engine.register_manager("object_manager", TrafficObjectManager())

