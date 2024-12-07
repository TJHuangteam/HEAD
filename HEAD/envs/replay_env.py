from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy, ReplayTrafficParticipantPolicy
from metadrive.policy.idm_policy import IDMPolicy


asset_path = AssetLoader.asset_path
env = ScenarioEnv({
    "use_render": True,
    "data_directory": AssetLoader.file_path(asset_path,"nuplan_pittsburgh", unix_style=False),
    "sequential_seed": True,
    # "manual_control":True,
    "show_sidewalk":True,
    "agent_policy":ReplayEgoCarPolicy,
    "num_scenarios": 100,
    "reactive_traffic": True,
    "horizon": 200
})
try:
    env.reset(seed=0)
    for t in range(10000):
        o, r, tm, tc, info = env.step([0, 0])
        # env.render(mode="top_down",
        #            window=True,
        #            screen_record=False,
        #            screen_size=(700,700))
        if tm or tc:
            env.reset()
finally:
    env.close()