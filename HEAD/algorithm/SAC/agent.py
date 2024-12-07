#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2021-04-29 12:53:54
LastEditor: JiangJi
LastEditTime: 2021-04-29 13:56:39
Discription: 
Environment: 
'''
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.Algo.common.memory import ReplayBuffer
from src.Algo.SAC.model import ValueNet, PolicyNet, SoftQNet
import math


class SAC:
    def __init__(self, state_dim, action_dim, cfg) -> None:
        self.alpha = 1.
        self.batch_size = cfg.batch_size
        self.memory = ReplayBuffer(cfg.capacity)
        self.device = cfg.device
        self.soft_q_net1 = SoftQNet(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.soft_q_net2 = SoftQNet(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.target_soft_q_net1 = SoftQNet(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.target_soft_q_net2 = SoftQNet(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.policy_net = PolicyNet(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=self.device)
        # print('Soft Q Network (1,2): ', self.soft_q_net1)
        # print('Policy Network: ', self.policy_net)

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=cfg.soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=cfg.soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.returns = 0

    def update(self, reward_scale=1., auto_entropy=True, target_entropy=-1, gamma=0.99, soft_tau=1e-2):

        if len(self.memory) < self.batch_size:
            return

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)


        state = torch.FloatTensor(np.array(state)).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(dim=-1).to(self.device)
        done = torch.FloatTensor(np.float32(np.array(done))).unsqueeze(dim=-1).to(self.device)

        # normalize reward
        # print(state)
        # reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(
        #     dim=0) + 1e-6)  # normalize with batch mean and std; plus a small number to prevent numerical problem

        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)

        # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q)
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        # Training Q Function
        target_q_min = torch.min(self.target_soft_q_net1(next_state, new_next_action),
                                 self.target_soft_q_net2(next_state, new_next_action)) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * gamma * target_q_min  # if done==1, only reward
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1,
                                               target_q_value.detach())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()

        # Training Policy Function
        predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action), self.soft_q_net2(state, new_action))
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # print('q loss: ', q_value_loss1, q_value_loss2)
        # print('policy loss: ', policy_loss )

        # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

    def save(self, path):
        torch.save(self.soft_q_net1.state_dict(), path + 'sac_soft_q1')
        torch.save(self.soft_q_optimizer1.state_dict(), path + "sac_soft_q1_optimizer")

        torch.save(self.soft_q_net2.state_dict(), path + 'sac_soft_q2')
        torch.save(self.soft_q_optimizer2.state_dict(), path + "sac_soft_q2_optimizer")

        torch.save(self.policy_net.state_dict(), path + "sac_policy")
        torch.save(self.policy_optimizer.state_dict(), path + "sac_policy_optimizer")

    def load(self, path):
        self.soft_q_net1.load_state_dict(torch.load(path + 'sac_soft_q1'))
        self.soft_q_optimizer1.load_state_dict(torch.load(path + "sac_soft_q1_optimizer"))

        self.soft_q_net2.load_state_dict(torch.load(path + 'sac_soft_q2'))
        self.soft_q_optimizer2.load_state_dict(torch.load(path + "sac_soft_q2_optimizer"))

        self.policy_net.load_state_dict(torch.load(path + "sac_policy"))
        self.policy_optimizer.load_state_dict(torch.load(path + "sac_policy_optimizer"))
