# -*- coding: utf-8 -*-
import os
import time
import gym
import numpy as np
import matplotlib.pyplot as plt

env_name = 'Swimmer-v2'
env = gym.make(env_name)

print('Observation space: ', env.observation_space)
print('Action space: ', env.action_space)
print('Observation space low: ', env.observation_space.low)
print('Observation space high: ', env.observation_space.high)
print('Action space low: ', env.action_space.low)
print('Action space high: ', env.action_space.high)

state = env.reset()
for _ in range(1000):
    env.render()
    print(env.sim.data.qpos[0])
    state, reward, done, _ = env.step(env.action_space.sample())
env.close()
