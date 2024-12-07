import time

from HEAD.adversarial_loop.envs.SAC_env import SACEnv
from HEAD.algorithm.SAC.SAC_learner import SACConfig
from HEAD.algorithm.SAC.agent import SAC
from HEAD.manager.traffic_manager import traffic_manager
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.policy.idm_policy import IDMPolicy
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os

from HEAD.adversarial_loop.policy.cut_in_policy import CutInPolicy

def train(env, RL_agent, SAC_cfg, writer):
    print('Start to train !')
    rewards = []
    total_nums = 0
    for i_ep in range(SAC_cfg.train_eps):
        obs, info = env.reset()
        cfg = env.config["vehicle_config"]
        cfg["agent_policy"] = IDMPolicy
        # other_traffic_vehicle_ids = setup_adv_traffic_vehicles(env)
        # manager = traffic_manager(env)
        # other_traffic_vehicle_ids = manager.setup_adv_traffic_vehicles()

        filter = []
        # filter.extend(other_traffic_vehicle_ids)
        v = env.engine.spawn_object(DefaultVehicle,
                                    vehicle_config=cfg,
                                    position=[0, 3.5],
                                    heading=0.0)
        filter.append(v.id)
        policy = IDMPolicy(v, env.current_seed)
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        eps_reward = 0.0
        for i_step in range(SAC_cfg.train_steps):
            total_nums = total_nums + 1
            action = RL_agent.policy_net.get_action(obs)
            act_action = np.append(0, action)
            print('action:', act_action)
            next_obs, reward, terminated, truncated, info = env.step(act_action)
            # manager.tick()

            v_action = policy.act()
            v.before_step(v_action)

            done = terminated or truncated
            RL_agent.memory.push(obs, action, reward, next_obs, done)
            RL_agent.update(reward_scale = 1., auto_entropy = SAC_cfg.AUTO_ENTROPY,
                                      target_entropy = -1. * env.action_space.shape[0], gamma = SAC_cfg.gamma,
                                      soft_tau = SAC_cfg.soft_tau)
            obs = next_obs
            eps_reward += reward
            if done:
                break
        mean_reward = eps_reward / (i_step + 1)

        rewards.append(mean_reward)
        print(f"Episode:{i_ep + 1}/{SAC_cfg.train_eps}, Reward:{mean_reward:.3f}")
        print(f'Total steps：{total_nums}')
        writer.add_scalar("Reward", mean_reward, total_nums)
        env.engine.clear_objects(filter, True)
    print('Complete training！')
    env.close()
    RL_agent.save(path='./Results/RL_Results/model_save/' + SAC_cfg.train_name + "/")

def continue_train(env, model_path, SAC_cfg, writer, continued, save_by_step=None):
    global total_nums, finished_episodes, last_mean_reward, RL_agent
    if continued:
        if os.path.exists(model_path + 'training_info'):  # Import the trained model, or start training from scratch if it is not available
            training_info = RL_agent.load(model_path)
            total_nums = training_info['total_nums']
            finished_episodes = training_info['finished_episodes']
        else:
            print('没有预训练模型，将从零开始训练')
            print('3')
            time.sleep(1)
            print('2')
            time.sleep(1)
            print('1')
            time.sleep(1)
    print('Start to train !')
    rewards = []
    for i_ep in range(finished_episodes, SAC_cfg.train_eps):
        obs, info = env.reset()
        cfg = env.config["vehicle_config"]
        cfg["agent_policy"] = IDMPolicy
        cfg['agent_policy'].enable_lane_change = False
        # other_traffic_vehicle_ids = setup_adv_traffic_vehicles(env)
        # manager = traffic_manager(env)
        # other_traffic_vehicle_ids = manager.setup_adv_traffic_vehicles()

        filter = []
        # filter.extend(other_traffic_vehicle_ids)
        v = env.engine.spawn_object(DefaultVehicle,
                                    vehicle_config=cfg,
                                    position=[0, 3.5],
                                    heading=0.0)
        filter.append(v.id)
        policy = IDMPolicy(v, env.current_seed)
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        eps_reward = 0.0
        for i_step in range(SAC_cfg.train_steps):
            total_nums = total_nums + 1
            action = RL_agent.policy_net.get_action(obs)
            act_action = np.append(0, action)
            print('action:', act_action)
            next_obs, reward, terminated, truncated, info = env.step(act_action)
            # manager.tick()

            v_action = policy.act()
            v.before_step(v_action)

            done = terminated or truncated
            RL_agent.memory.push(obs, action, reward, next_obs, done)
            RL_agent.update(reward_scale = 1., auto_entropy = SAC_cfg.AUTO_ENTROPY,
                                      target_entropy = -1. * env.action_space.shape[0], gamma = SAC_cfg.gamma,
                                      soft_tau = SAC_cfg.soft_tau)
            obs = next_obs
            eps_reward += reward
            if save_by_step is not None:
                if total_nums % save_by_step == 0:
                    training_info = {
                        'total_nums': total_nums,
                        'finished_episodes': finished_episodes,
                        'last_mean_reward': last_mean_reward
                    }
                    RL_agent.save_by_step(path='./Results/RL_Results/model_save/' + SAC_cfg.train_name + "/", training_info=training_info)
            if done:
                break
        mean_reward = eps_reward / (i_step + 1)

        rewards.append(mean_reward)
        print(f"Episode:{i_ep + 1}/{SAC_cfg.train_eps}, Reward:{mean_reward:.3f}")
        print(f'Total steps：{total_nums}')
        writer.add_scalar("Reward", mean_reward, total_nums)
        env.engine.clear_objects(filter, True)
        finished_episodes += 1
        last_mean_reward = mean_reward
    print('Complete training！')
    env.close()
    training_info = {
        'total_nums': total_nums,
        'finished_episodes': finished_episodes,
        'last_mean_reward': last_mean_reward
    }
    RL_agent.save(path='./Results/RL_Results/model_save/' + SAC_cfg.train_name + "/", training_info=training_info)

def eval(env, RL_agent, model_path, SAC_cfg, writer):
    print('Start to eval !')
    rewards = []
    total_nums = 0
    RL_agent.load(model_path)
    for i_ep in range(SAC_cfg.eval_eps):
        obs, info = env.reset()
        cfg = env.config["vehicle_config"]
        cfg["agent_policy"] = IDMPolicy
        cfg['agent_policy'].enable_lane_change = False
        filter = []
        frames = []
        v = env.engine.spawn_object(DefaultVehicle,
                                    vehicle_config=cfg,
                                    position=[0, 3.5],
                                    heading=0.0)
        filter.append(v.id)

        policy = IDMPolicy(v, env.current_seed)
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        eps_reward = 0.0
        for i_step in range(SAC_cfg.eval_steps):
            total_nums = total_nums + 1
            action = RL_agent.policy_net.get_eval_action(obs)
            act_action = np.append(0, action)
            print('action:', act_action)
            next_obs, reward, terminated, truncated, info = env.step(act_action)
            v_action = policy.act()
            v.before_step(v_action)  #
            done = terminated or truncated
            obs = next_obs
            eps_reward += reward
            # frame = env.render(mode="topdown",
            #                    window=False,
            #                    screen_size=(1080, 720),
            #                    camera_position=(150, 10.5 + 3.5/2))
            # frames.append(frame)
            if done:
                break
        mean_reward = eps_reward / (i_step + 1)
        # generate_gif(frames, gif_name=f"{i_ep}.gif")

        rewards.append(mean_reward)
        print(f"Episode:{i_ep + 1}/{SAC_cfg.eval_eps}, Reward:{mean_reward:.3f}")
        print(f'总步数：{total_nums}')
        env.engine.clear_objects(filter, True)
    print('Complete eval！')
    env.close()


if __name__ == '__main__':
    train_name = "train_11"
    total_nums = 0
    finished_episodes = 0
    last_mean_reward = 0
    SAC_cfg = SACConfig('self-learning', train_name)
    state_dim = 259
    action_dim = 1
    RL_agent = SAC(state_dim, action_dim, SAC_cfg)
    env = SACEnv()

    env.config['agent_policy'] = CutInPolicy
    env.config['agent_configs']['default_agent']['spawn_longitude'] = 2
    model_path = './Results/RL_Results/model_save/' + train_name +'/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    log_path = './Results/RL_Results/runs_info/' + train_name +'/'
    eval_path = './Results/RL_Results/eval_info/' + train_name +'/'

    train_or_eval = False
    if train_or_eval:
        try:
            writer = SummaryWriter(log_path)
            # continue_train(env, model_path, SAC_cfg, writer, continued=True, save_by_step=500)
            continue_train(env, model_path, SAC_cfg, writer, continued=True, save_by_step=None)
        except KeyboardInterrupt:
            writer.add_scalar("Reward", last_mean_reward, total_nums)
            writer.close()
            training_info = {
                'total_nums': total_nums,
                'finished_episodes': finished_episodes,
                'last_mean_reward': last_mean_reward
            }
            RL_agent.save(path='./Results/RL_Results/model_save/' + SAC_cfg.train_name + "/", training_info=training_info)

    else:
        writer = SummaryWriter(eval_path)
        eval(env, RL_agent, model_path, SAC_cfg, writer)

