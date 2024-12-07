# -*-coding:utf-8-*-
import os
import sys
import time
import re
import gym
import torch
import datetime
import csv
from src.Algo.SAC.env import NormalizedActions
from src.Algo.SAC.agent import SAC
from src.Algo.common.utils import save_results, make_dir
import numpy as np
from collections import deque

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import tkinter as tk

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
root_path = 'D:\DOWN LOAD\PyDownload\continual_scenario_algos -1'
# root_path = os.path.dirname(root_path)
sys.path.append(parent_path)  # 添加路径到系统路径


def get_dir_path(path):
    """
    创建指定的文件夹
    :param path: 文件夹路径，字符串格式
    :return: True(新建成功) or False(文件夹已存在，新建失败)
    """

    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        # os.makedirs(path)
        return path, None
    else:
        path, path_origin = directory_check(path)
        # os.makedirs(path)
        return path + '/', path_origin + '/'


def directory_check(directory_check):
    temp_directory_check = directory_check
    i = 1
    while i:

        if os.path.exists(temp_directory_check):
            search = '_'
            numList = [m.start() for m in re.finditer(search, temp_directory_check)]
            numList[-1]
            temp_directory_check = temp_directory_check[0:numList[-1] + 1] + str(i)
            i = i + 1
        else:
            return temp_directory_check, temp_directory_check[0:numList[-1] + 1] + str(i - 2)


class SACConfig:
    def __init__(self, args):
        self.algo = 'SAC'
        self.env_name = 'Metadrive'
        self.train_name = args.train_name

        self.model_save_path = root_path + "/Results/RL/Saved Models"
        self.log_save_path = root_path + "/Results/RL/runs_info/"
        self.eval_save_path = root_path + "/Results/RL/eval_info/"
        self.train_eps = 1000000
        self.eps_max_steps = 2000
        self.eval_eps = 1000
        self.total_steps = args.total_steps
        self.gamma = 0.99
        self.soft_tau = 5e-3
        self.soft_q_lr = 3e-4
        self.policy_lr = 3e-4
        self.capacity = 1000000

        self.eval_total_steps = 60000
        self.hidden_dim = 256
        self.batch_size = 256
        self.alpha_lr = 3e-4
        self.AUTO_ENTROPY = True
        self.DETERMINISTIC = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_flag = args.train_flag


class SAC_Learner:

    def __init__(self, env, SAC_cfg):

        self.env = env
        self.writer = None
        self.rewards = None
        self.ma_rewards = None
        self.SAC_cfg = SAC_cfg
        self.model_save_path = self.SAC_cfg.model_save_path
        self.model_path = self.model_save_path + '/' + self.SAC_cfg.train_name + '/'
        self.log_path = self.SAC_cfg.log_save_path + self.SAC_cfg.train_name + '/'
        self.eval_path = self.SAC_cfg.eval_save_path + '/' + self.SAC_cfg.train_name + '/'
        self.fps = 0.0

        if self.SAC_cfg.train_flag:
            self.model_path, _ = get_dir_path(self.model_path)

        self.agent = None
        self.curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # obtain current time

        self.ep_info_buffer = None
        self.ep_len = 0
        # todo:cpu
        # self.device = self.SAC_cfg.device

    def agent_initialize(self):

        print('Env is starting')
        action_dim = self.env.action_space.shape[0]
        state_dim = self.env.observation_space.shape[0]

        self.agent = SAC(state_dim, action_dim, self.SAC_cfg)


        print(self.SAC_cfg.algo + ' algorithm is starting')
        # tensorboard
        if self.SAC_cfg.train_flag:
            self.log_path, _ = get_dir_path(self.log_path)
            self.writer = SummaryWriter(self.log_path)  # default at runs folder if not sepecify path
        else:
            self.eval_path, _ = get_dir_path(self.eval_path)
            self.writer = SummaryWriter(self.eval_path)  # default at runs folder if not sepecify path

    def load(self):
        self.agent.load(self.model_path)
        print('agent ' + self.SAC_cfg.train_name + ' is loaded')

    def train(self):

        print('Start to train !')
        print(f'Env: {self.SAC_cfg.env_name}, Algorithm: {self.SAC_cfg.algo}, Device: {self.SAC_cfg.device}')
        total_nums = 0
        frame_count = 0
        self.ep_info_buffer = deque(maxlen=30)
        self.ep_len_buffer = deque(maxlen=30)
        start_time = time.time()
        ep_reward = np.zeros(self.env.num_envs)
        ep_len = np.zeros(self.env.num_envs)
        for i_ep in range(self.SAC_cfg.train_eps):
            state = self.env.reset()
            # todo:cpu
            # state = torch.tensor(state, device=self.device, dtype=torch.float32)

            reset_flag = np.array([True for i in range(self.env.num_envs)])
            for i_step in range(self.SAC_cfg.eps_max_steps):
                total_nums = total_nums + 1
                action = self.agent.policy_net.get_action(state)


                next_state, reward, done, info = self.env.step([row for row in action])
                # todo:cpu
                # next_state = torch.tensor(next_state, device=self.device, dtype=torch.float32)
                reward = torch.tensor(reward, device=self.device, dtype=torch.float32)

                for i in range(self.env.num_envs):
                    if reset_flag[i]:
                        frame_count += 1
                        self.agent.memory.push(state[i], action[i], reward[i], next_state[i], done[i])

                self.agent.update(reward_scale=1., auto_entropy=self.SAC_cfg.AUTO_ENTROPY,
                                  target_entropy=-1. * self.env.action_space.shape[0], gamma=self.SAC_cfg.gamma,
                                  soft_tau=self.SAC_cfg.soft_tau)

                state = next_state

                if (time.time() - start_time >= 1):
                    self.fps = frame_count / (time.time() - start_time)
                    frame_count = 0
                    start_time = time.time()

                for d in range(self.env.num_envs):
                    if done[d]:
                        reset_flag[d] = False
                        total_nums += info[d]['episode_length']
                        ep_len[d] = info[d]['episode_length']
                        ep_reward[d] = info[d]['episode_reward']
                if not reset_flag.any():
                    break

            self.ep_info_buffer.append(ep_reward[0])
            self.ep_len_buffer.append(ep_len[0])

            mean_eps_reward = np.mean([ep_info for ep_info in self.ep_info_buffer])
            mean_eps_len = np.mean([ep_len for ep_len in self.ep_len_buffer])
            print(
                f"Episode:{i_ep + 1}/{self.SAC_cfg.train_eps}, Reward:{mean_eps_reward:.3f}, fps:{self.fps}, ep_len: {mean_eps_len}")
            print(f'总步数：{total_nums}')
            self.writer.add_scalar("ep_rew_mean", mean_eps_reward, total_nums)
            self.writer.add_scalar("ep_len", mean_eps_len, total_nums)
            self.writer.add_scalar("fps", self.fps, total_nums)

            if total_nums >= self.SAC_cfg.total_steps:
                break

        print('Complete training！')

    def eval(self):
        print('Start to eval !')
        print(f'Env: {self.SAC_cfg.env_name}, Algorithm: {self.SAC_cfg.algo}, Device: {self.SAC_cfg.device}')
        rewards = []
        ma_rewards = []  # moveing average reward
        total_nums = 0
        success_list = []
        reward_all = 0

        for i_ep in range(self.SAC_cfg.eval_eps):

            state, _ = self.env.reset()

            eps_reward = 0.0
            ep_len = 0
            for i_step in range(self.SAC_cfg.eps_max_steps):

                total_nums = total_nums + 1
                action = self.agent.policy_net.get_eval_action(state)
                # action = [0.015, 0.015]
                # print(action)
                next_state, reward, done, termin, info = self.env.step(action)

                ret = self.env.render(mode="topdown",
                                      screen_record=True)

                state = next_state
                eps_reward += reward
                ep_len += 1
                if done or termin:
                    print("episode_len", ep_len)

                    ep_len = 0
                    break

            rewards.append(eps_reward)

            print(f"Episode:{i_ep + 1}/{self.SAC_cfg.eval_eps}, Reward:{eps_reward:.3f}")
            reward_all += eps_reward  # 总奖励
            mean_reward_all = reward_all / total_nums
            mean_reward = eps_reward / i_step
            print("整体reward per step:", mean_reward_all)
            print("当前回合reward per step:", mean_reward)

            success_list.append(info['arrive_dest'])
            success_rate = sum(success_list) / len(success_list)
            print(f'成功率：{success_rate * 100:.2f}%，（{sum(success_list)} / {len(success_list)}）')
            print(f'总步数：{total_nums}')
            self.writer.add_scalar("ep_rew_mean", eps_reward, total_nums)
            self.writer.add_scalar("reward per step", mean_reward, total_nums)  #怎么衡量一下reward per step

            if total_nums >= self.SAC_cfg.total_steps:
                break
        print('Complete evaluating')
        return rewards, ma_rewards

    def save(self):
        make_dir(self.model_path)
        self.agent.save(path=self.model_path)
