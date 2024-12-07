import copy
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from typing import Union

import numpy as np
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.algorithm.blocks_prob_dist import PGBlockDistConfig
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import parse_map_config, MapGenerateMethod
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.constants import DEFAULT_AGENT, TerminationState
from metadrive.envs.base_env import BaseEnv
from metadrive.manager.traffic_manager import TrafficMode
from metadrive.utils import clip, Config
import os

size = (256, 128) if not os.getenv('TEST_DOC') else (16, 16) # for github CI

METADRIVE_DEFAULT_CONFIG = dict(
    # ===== Generalization =====
    start_seed=0,
    num_scenarios=1,

    # ===== PG Map Config =====
    map = "SSS" ,  # int or string: an easy way to fill map_config, three straight blocks
    block_dist_config=PGBlockDistConfig,
    random_lane_width=False,
    random_lane_num=False,
    map_config={
        BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
        BaseMap.GENERATE_CONFIG: None,  # it can be a file path / block num / block ID sequence
        BaseMap.LANE_WIDTH: 3.5,
        BaseMap.LANE_NUM: 3,
        "exit_length": 50,
    },
    store_map=True,

    # ===== Traffic =====
    traffic_density=0.1,
    need_inverse_traffic=False,
    traffic_mode=TrafficMode.Trigger,  # "Respawn", "Trigger"
    random_traffic=False,  # Traffic is randomized at default.
    # this will update the vehicle_config and set to traffic
    traffic_vehicle_config=dict(
        show_navi_mark=False,
        show_dest_mark=False,
        enable_reverse=False,
        show_lidar=False,
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
    vehicle_config=dict(navigation_module=NodeNetworkNavigation),
    agent_configs={
        DEFAULT_AGENT: dict(
            use_special_color=True,
            spawn_lane_index=(FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0),
        )
    },

    # ===== Reward Scheme =====
    # See: https://github.com/metadriverse/metadrive/issues/283
    success_reward=10.0,
    out_of_road_penalty=5.0,
    crash_vehicle_penalty=5.0,
    crash_object_penalty=5.0,
    driving_reward=1.0,
    speed_reward=0.1,
    use_lateral_reward=False,

    # ===== Cost Scheme =====
    crash_vehicle_cost=1.0,
    crash_object_cost=1.0,
    out_of_road_cost=1.0,

    # ===== Termination Scheme =====
    out_of_route_done=False,
    on_continuous_line_done=True,
    crash_vehicle_done=True,
    crash_object_done=True,
    crash_human_done=True,

    sensors=dict(rgb_camera=[RGBCamera, *size])
)


class MetaDriveEnv(BaseEnv):
    @classmethod
    def default_config(cls) -> Config:
        config = super(MetaDriveEnv, cls).default_config()
        config.update(METADRIVE_DEFAULT_CONFIG)
        config.register_type("map", str, int)
        config["map_config"].register_type("config", None)
        return config

    def __init__(self, config: Union[dict, None] = None):
        self.default_config_copy = Config(self.default_config(), unchangeable=True)
        super(MetaDriveEnv, self).__init__(config)

        # scenario setting
        self.start_seed = self.start_index = self.config["start_seed"]
        self.env_num = self.num_scenarios = self.config["num_scenarios"]

    def _post_process_config(self, config):
        config = super(MetaDriveEnv, self)._post_process_config(config)
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
        done = False
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
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: arrive_dest.".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.OUT_OF_ROAD]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: out_of_road.".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_VEHICLE] and self.config["crash_vehicle_done"]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: crash vehicle ".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_OBJECT] and self.config["crash_object_done"]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: crash object ".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_BUILDING]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: crash building ".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_HUMAN] and self.config["crash_human_done"]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: crash human".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.MAX_STEP]:
            # single agent horizon has the same meaning as max_step_per_agent
            if self.config["truncate_as_terminate"]:
                done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: max step ".format(self.current_seed),
                extra={"log_once": True}
            )
        return done, done_info

    def cost_function(self, vehicle_id: str): # 成本最小化 和 奖励最大化 ；cost func是一个优化方向，主要保障安全性
        vehicle = self.agents[vehicle_id]
        step_info = dict()
        step_info["cost"] = 0
        if self._is_out_of_road(vehicle): # 进入判断，是否vehicle发生以下行为
            step_info["cost"] = self.config["out_of_road_cost"]
        elif vehicle.crash_vehicle:
            step_info["cost"] = self.config["crash_vehicle_cost"]
        elif vehicle.crash_object:
            step_info["cost"] = self.config["crash_object_cost"]
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
        # long，lat代表frenet坐标系下的横纵向位置；每个object下设navigation信息，local）coordinates是转换坐标函数
        flag = (vehicle.navigation.final_lane.length - 5 < long < vehicle.navigation.final_lane.length + 5) and (
            vehicle.navigation.get_current_lane_width() / 2 >= lat >=
            (0.5 - vehicle.navigation.get_current_lane_num()) * vehicle.navigation.get_current_lane_width()
        ) # 判断是否在destination范围区间
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
        计算指定车辆的奖励值，基于其动作和位置。
        可以重写该函数来修改奖励结构。

        :param vehicle_id: 环境中车辆（智能体）的标识符。
        :return: 为该车辆计算的奖励值。
        """
        vehicle = self.agents[vehicle_id]  # 根据车辆ID获取智能体（车辆）对象
        step_info = dict()  # 字典，用于存储每一步的奖励相关信息

        # 检查车辆是否在其参考车道（目标或分配车道）内
        # 参考车道定义为车辆当前应该遵循的车道集合
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane  # 将当前车道设为车辆所在车道
            positive_road = 1  # 表示车辆在正确的车道上
        else:
            # 如果不在参考车道内，将第一个参考车道作为目标车道
            current_lane = vehicle.navigation.current_ref_lanes[0]
            current_road = vehicle.navigation.current_road
            # 检查当前道路是否为反向车道
            positive_road = 1 if not current_road.is_negative_road() else -1

        # 计算车辆在当前车道内的纵向和横向位置
        long_last, _ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

        # 对保持车辆在车道内的奖励：
        # 该部分对横向偏离进行惩罚，以鼓励车辆保持在车道范围内
        if self.config["use_lateral_reward"]:
            # 基于离车道中心的横向距离的奖励因子
            lateral_factor = clip(1 - 2 * abs(lateral_now) / vehicle.navigation.get_current_lane_width(), 0.0, 1.0)
        else:
            lateral_factor = 1.0  # 如果禁用横向奖励，将横向因子设为1（无惩罚）

        # 初始化奖励值
        reward = 0.0
        # 对在车道内前进的奖励（与纵向移动的距离成比例）
        reward += self.config["driving_reward"] * (long_now - long_last) * lateral_factor * positive_road
        # 对保持速度的奖励（与车辆速度与最大速度的比例相关）
        reward += self.config["speed_reward"] * (vehicle.speed_km_h / vehicle.max_speed_km_h) * positive_road

        step_info["step_reward"] = reward  # 存储每一步的奖励信息

        # 可以在此处处理附加条件，例如到达目的地、冲出路面、碰撞......
        if self._is_arrive_destination(vehicle):
            reward = +self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward = -self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            reward = -self.config["crash_object_penalty"]
        return reward, step_info

    def setup_engine(self):
        """
        设置环境的引擎，包括地图、交通和对象管理器。
        """
        super(MetaDriveEnv, self).setup_engine()  # 调用父类的 setup_engine 方法，初始化引擎
        # 导入所需的管理器模块
        from metadrive.manager.traffic_manager import PGTrafficManager
        from metadrive.manager.pg_map_manager import PGMapManager
        from metadrive.manager.object_manager import TrafficObjectManager
        # 注册地图管理器，用于生成和管理地图
        self.engine.register_manager("map_manager", PGMapManager())
        # 注册交通管理器，用于管理交通流和车辆行为
        self.engine.register_manager("traffic_manager", PGTrafficManager())
        # 如果事故概率不为零，注册对象管理器，用于管理交通对象，如障碍物等
        if abs(self.config["accident_prob"] - 0) > 1e-2:
            self.engine.register_manager("object_manager", TrafficObjectManager())


if __name__ == '__main__':

    def _act(env, action):
        assert env.action_space.contains(action)
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert np.isscalar(reward)
        assert isinstance(info, dict)

    env = MetaDriveEnv()
    try:
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        _act(env, env.action_space.sample())
        for x in [-1, 0, 1]:
            env.reset()
            for y in [-1, 0, 1]:
                _act(env, [x, y])
    finally:
        env.close()
