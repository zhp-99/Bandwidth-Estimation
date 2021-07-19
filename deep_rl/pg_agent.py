#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import time
from torch import nn
from torch.distributions import MultivariateNormal


class A2C(nn.Module):
    def __init__(self, state_dim, action_dim, lr=1e-3, betas=(0.9, 0.999), gamma=0.99, A2C_epoch=0):
        super(A2C, self).__init__()
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.A2C_epoch = A2C_epoch

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device('cpu')

        self.actor_critic = ActorCritic(state_dim, action_dim).to(self.device)
        self.actor_critic_optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr, betas=betas)

    def select_action(self, state, storage):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action, action_logprobs, entropy, value = self.actor_critic(state)

        storage.logprobs.append(action_logprobs)
        storage.values.append(value)
        storage.states.append(state)
        storage.actions.append(action)
        storage.entropy += entropy
        return action

    def forward(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action, action_logprobs, entropy, value = self.actor_critic(state)
        return action, action_logprobs, value

    def get_value(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor_critic.critic(state)

    def update(self, storage, state):
        episode_policy_loss = 0
        episode_value_loss = 0

        log_probs = torch.cat(storage.logprobs).to(self.device)
        returns = torch.cat(storage.returns).detach().to(self.device)
        values = torch.cat(storage.values).to(self.device)
        entropy = storage.entropy.to(self.device)

        advantage = returns - values

        #print(log_probs.shape,returns.shape,values.shape,advantage.shape)


        # advantages = (torch.tensor(storage.returns) - torch.tensor(storage.values)).detach()
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        #
        # old_states = torch.squeeze(torch.stack(storage.states).to(self.device), 1).detach()
        # old_actions = torch.squeeze(torch.stack(storage.actions).to(self.device), 1).detach()
        # old_action_logprobs = torch.squeeze(torch.stack(storage.logprobs), 1).to(self.device).detach()
        # old_returns = torch.squeeze(torch.stack(storage.returns), 1).to(self.device).detach()

        # for t in range(self.A2C_epoch):
            # logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            # ratios = torch.exp(logprobs - old_action_logprobs)
            #
            # surr1 = ratios * advantages
            # surr2 = torch.clamp(ratios, 1-self.ppo_clip, 1+self.ppo_clip) * advantages
            # policy_loss = -torch.min(surr1, surr2).mean()
            # value_loss = 0.5 * (state_values - old_returns).pow(2).mean()
            # #loss = policy_loss + value_loss
            #
            # self.actor_optimizer.zero_grad()
            # policy_loss.backward()
            # self.actor_optimizer.step()
            #
            # self.critic_optimizer.zero_grad()
            # value_loss.backward()n
            # self.critic_optimizer.step()


        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + critic_loss - 0.00*entropy

        self.actor_critic_optimizer.zero_grad()
        loss.backward()
        self.actor_critic_optimizer.step()

        episode_policy_loss += actor_loss.detach()
        episode_value_loss += critic_loss.detach()


        return episode_policy_loss / self.A2C_epoch, episode_value_loss / self.A2C_epoch

    def save_model(self, data_path, epoch):
        #assert 1==0
        checkpoint = {
            'net':self.actor_critic.state_dict(),
            'optimizer':self.actor_critic_optimizer.state_dict(),
            'epoch':epoch
        }
        torch.save(checkpoint, '{}a2c_{}.pth'.format(data_path, epoch))

class RNN_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64, num_layers=3):
        super(RNN_Actor, self).__init__()
        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=hidden_size, bias=True, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size, action_dim)
        self.action_var = torch.full((action_dim,), 0.05 ** 2)

    def forward(self, x):
        device = x.device
        x = self.linear(x)

        #mean,std in [0,1]
        mean = self.mean_linear(x)
        cov_mat = torch.diag(self.action_var).to(device)

        # print(mean)
        # print(std)

        dist = MultivariateNormal(mean, cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()
        return action, log_prob, entropy


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128,64)
        )
        self.mean_linear = nn.Sequential(
            nn.Linear(64,action_dim),
            nn.Sigmoid()
        )
        self.std_linear = nn.Sequential(
            nn.Linear(64, action_dim),
        )

        self.action_var = torch.full((action_dim,), 0.05 ** 2)

    def forward(self, x):
        device = x.device
        x = self.linear(x)

        #mean,std in [0,1]
        mean = self.mean_linear(x)
        cov_mat = torch.diag(self.action_var).to(device)

        # print(mean)
        # print(std)

        dist = MultivariateNormal(mean, cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()
        return action, log_prob, entropy

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
    def forward(self, x):
        x = self.linear(x)
        return x.view(1)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic,self).__init__()
        self.actor = Actor(state_dim,action_dim)
        self.critic = Critic(state_dim)

    def forward(self, x):
        action, action_logprob, entropy = self.actor(x)
        value = self.critic(x)
        return action, action_logprob, entropy, value