#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import torch
import matplotlib.pyplot as plt
import matplotlib

import draw
from rtc_env import GymEnv
from deep_rl.storage import Storage
from deep_rl.pg_agent import A2C
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default=None)
FLAGS = parser.parse_args()

checkpoint_path = FLAGS.checkpoint_path

#device = torch.device("cuda")# if torch.cuda.is_available() else "cpu")

def main():
    #画图仅保存，不显示
    matplotlib.use('Agg')

    ############## Hyperparameters for the experiments ##############
    env_name = "AlphaRTC"
    max_num_episodes = 50      # maximal episodes

    update_interval = 4000      # update policy every update_interval timesteps
    save_interval = 2          # save model every save_interval episode
    exploration_param = 0.05    # the std var of action distribution
    K_epochs = 37               # update policy for K_epochs
    ppo_clip = 0.2              # clip parameter of PPO
    gamma = 0.99                # discount factor

    lr = 3e-3                 # Adam parameters
    betas = (0.9, 0.999)
    state_dim = 4
    action_dim = 1
    data_path = f'./data/' # Save model and reward curve here
    #############################################

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    env = GymEnv()
    storage = Storage() # used for storing data
    model = A2C(state_dim, action_dim, lr, betas, gamma, K_epochs)#.to(device)

    record_episode_reward = []
    episode_reward  = 0
    time_step = 0

    # training loop
    start_epoch = 0
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.actor_critic.load_state_dict(checkpoint['net'])
        model.actor_critic_optimizer.load_state_dict((checkpoint['optimizer']))
        start_epoch = checkpoint['epoch'] + 1

    for episode in range(start_epoch, start_epoch + max_num_episodes):
        while time_step < update_interval:
            done = False
            state = torch.Tensor(env.reset())
            while not done and time_step < update_interval:
                action = model.select_action(state, storage)
                action = action.cpu()
                state, reward, done, _ = env.step(action)
                state = torch.Tensor(state)
                print(state.shape)
                input()
                # Collect data for update
                storage.rewards.append(reward)
                storage.is_terminals.append(done)
                time_step += 1
                episode_reward += reward

        next_value = model.get_value(state)
        storage.compute_returns(next_value, gamma)

        # update
        policy_loss, val_loss = model.update(storage, state)
        storage.clear_storage()
        episode_reward /= time_step
        record_episode_reward.append(episode_reward)
        print('Episode {} \t Average policy loss, value loss, reward {}, {}, {}'.format(episode, policy_loss, val_loss, episode_reward))

        if episode > 0 and not (episode % save_interval):
            model.save_model(data_path, episode)
            plt.plot(range(len(record_episode_reward)), record_episode_reward)
            plt.xlabel('Episode')
            plt.ylabel('Averaged episode reward')
            plt.savefig('%sreward_record.jpg' % (data_path))

        episode_reward = 0
        time_step = 0

    draw.draw_module(model.actor_critic, data_path)


if __name__ == '__main__':
    main()
