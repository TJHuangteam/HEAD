import numpy as np
from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.engine.logger import get_logger
from metadrive.component.vehicle.PID_controller import PIDController

from metadrive.utils.math import wrap_to_pi

logger = get_logger()
class CutInPolicy(EnvInputPolicy):
    def __init__(self, obj, seed):
        # Since control object may change
        super(CutInPolicy, self).__init__(obj, seed)
        # self.discrete_action = self.engine.global_config["discrete_action"]
        self.discrete_action = True
        assert self.discrete_action, "Must set discrete_action=True for using this control policy"
        self.use_multi_discrete = self.engine.global_config["use_multi_discrete"]
        self.steering_unit = 1.0
        self.throttle_unit = 2.0 / (
                self.engine.global_config["discrete_throttle_dim"] - 1
        )  # for discrete actions space
        self.discrete_steering_dim = 3  # only left or right
        self.discrete_throttle_dim = self.engine.global_config["discrete_throttle_dim"]
        logger.info("The discrete_steering_dim for {} is set to 3 [left, keep, right]".format(self.name))

        self.heading_pid = PIDController(1.7, 0.01, 3.5)
        self.lateral_pid = PIDController(0.3, .002, 0.05)

    def act(self, agent_id):
        action = self.engine.external_actions[agent_id]
        throttle = action[1]
        # if self.engine.global_config["action_check"]:
        #     # Do action check for external input in EnvInputPolicy
        #     assert self.get_input_space().contains(action), "Input {} is not compatible with action space {}!".format(
        #         action, self.get_input_space()
        #     )
        # to_process = self.convert_to_continuous_action(action) if self.discrete_action else action
        #
        # # clip to -1, 1
        # action = [clip(to_process[i], -1.0, 1.0) for i in range(len(to_process))]
        # self.action_info["action"] = action
        all_objects = self.engine.get_objects()
        filter = list(all_objects.keys())
        adv = all_objects[filter[0]]
        ego = all_objects[filter[1]]
        current_lane = adv.navigation.current_lane
        target_lane = current_lane

        if (adv.last_position[0] - ego.last_position[0]) > 4:  # 切入的纵向条件
            if adv.navigation.current_lane.index[-1] - ego.navigation.current_lane.index[-1] == -1:  # 对抗车在主车左侧切入
                lane_num = len(adv.navigation.current_ref_lanes)
                target_lane = adv.navigation.current_ref_lanes[min(current_lane.index[-1] + 1, lane_num - 1)]
            elif current_lane.index[-1] - ego.navigation.current_lane.index[-1] == 1:  # 对抗车在主车右侧切入
                target_lane = adv.navigation.current_ref_lanes[max(current_lane.index[-1] - 1, 0)]

        # elif -5 < (adv.last_position[0] - ego.last_position[0]) < -1:  # 超车的纵向条件
        #     if adv.navigation.current_lane.index[-1] - ego.navigation.current_lane.index[-1] == 0:  # 换道超车的横向条件
        #         target_lane = adv.navigation.current_ref_lanes[max(current_lane.index[-1] - 1, 0)]

        action = [self.steering_control(target_lane), throttle]
        self.action_info["action"] = action
        return action

    def steering_control(self, target_lane) -> float:
        # heading control following a lateral distance control
        ego_vehicle = self.control_object
        long, lat = target_lane.local_coordinates(ego_vehicle.position)
        lane_heading = target_lane.heading_theta_at(long + 1)
        v_heading = ego_vehicle.heading_theta
        steering = self.heading_pid.get_result(-wrap_to_pi(lane_heading - v_heading))
        steering += self.lateral_pid.get_result(-lat)
        return float(steering)
