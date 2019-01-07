# -*- coding: utf-8 -*-
import os
import time
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.classic_control.cartpole import *
from gym.envs.mujoco import *

# from gym.envs.registration import *
# from gym.wrappers.time_limit import TimeLimit

# env_name = 'Walker2d-v2'
# env = gym.make(env_name)
#
# print('Observation space: ', env.observation_space)
# print('Action space: ', env.action_space)
# print('Observation space low: ', env.observation_space.low)
# print('Observation space high: ', env.observation_space.high)
# print('Action space low: ', env.action_space.low)
# print('Action space high: ', env.action_space.high)
#
# state = env.reset()
# print(state)
# for _ in range(1000):
#     env.render()
#     time.sleep(0.01)
#     state, reward, done, _ = env.step(env.action_space.sample())
# env.close()


##### Render environment for a specific state #####
#
# env = CartPoleEnv()
# env.state = np.array([0.03971514, -0.01205, 0.039588, -0.00371212])
# # state = env.reset()
# while True:
#     env.render()

env = Walker2dEnv()

state = env.reset()
env.step(np.random.rand(6))
print('qpos: ', env.sim.data.qpos)
print('qvel: ', env.sim.data.qvel)
print('state: ', env._get_obs())
