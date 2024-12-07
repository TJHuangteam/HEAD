from importlib import import_module


class EnvironmentInfoCollector:
    def __init__(self, env, vehicle,config_dict):
        self._obs_managers = {}
        self._obs_configs = config_dict
        self.env = env
        self.vehicle = vehicle
        self._init_obs_manager()

    def get_observation(self):
        # 在这里添加你需要的代码来收集环境信息
        obs_dict = {}
        for ev_id, om_dict in self._obs_managers.items():### ev_id = vehicle_id om_dict = {speed gnss central_rgb route_plan birdview}
            obs_dict[ev_id] = {}
            for obs_id, om in om_dict.items():## speed ={speed speedxy forwardspeed...}
                obs_dict[ev_id][obs_id] = om.get_observation(self.vehicle, self.env)
        return obs_dict

    def _init_obs_manager(self):
        for ev_id, ev_obs_configs in self._obs_configs.items():
            self._obs_managers[ev_id] = {}
            for obs_id, obs_config in ev_obs_configs.items():
                ObsManager = getattr(import_module('HEAD.core.obs_manager.'+obs_config["module"]), 'ObsManager')
                self._obs_managers[ev_id][obs_id] = ObsManager(obs_config)

