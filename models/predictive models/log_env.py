# -*- coding: utf-8 -*-
import os
import time
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.classic_control.cartpole import *
from gym.envs.mujoco import *
import pandas as pd

# from gym.envs.registration import *
# from gym.wrappers.time_limit import TimeLimit

env_name = 'Walker2d-v2'
env = gym.make(env_name)

# print('Observation space: ', env.observation_space)
# print('Action space: ', env.action_space)
# print('Observation space low: ', env.observation_space.low)
# print('Observation space high: ', env.observation_space.high)
# print('Action space low: ', env.action_space.low)
# print('Action space high: ', env.action_space.high)

state = env.reset()
print(state)
while True:
    env.render()
    time.sleep(1)
    state, reward, done, _ = env.step(env.action_space.sample())
env.close()


##### Render environment for a specific state #####
#
# env = CartPoleEnv()
# env.state = np.array([0.03971514, -0.01205, 0.039588, -0.00371212])
# # state = env.reset()
# while True:
#     env.render()

# env = Walker2dEnv()
#
# state = env.reset()
# env.step(np.random.rand(6))
# print('qpos: ', env.sim.data.qpos)
# print('qvel: ', env.sim.data.qvel)
# print('state: ', env._get_obs())

# file_name_gt = 'GroundTruth_2019-01-11_20-01-56.csv'
# file_name_pred = 'Prediction_2019-01-11_20-01-56.csv'
# env_label = 'CartPole'
# df_test = pd.read_csv('./GN v1.5/results/' + env_label + '/test/' + file_name_pred, sep=',', header=0)
# env = Walker2dEnv()
# df = df_test.as_matrix()
# i = 300
# # _,_,s2,s3,s0,s1,_,_ = df[i,:]
# # env.state = np.array([s0,s1,s2,s3])
# while True:
#                       env.render()
