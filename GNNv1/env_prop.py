# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 12:35:23 2018

@author: ocliv
"""
import os
import time
import gym
import numpy as np
import matplotlib.pyplot as plt


env_name = 'MountainCarContinuous-v0'
env = gym.make(env_name)

print('Observation space: ', env.observation_space)
print('Action space: ', env.action_space)

print('Observation space low: ', env.observation_space.low)
print('Observation space high: ', env.observation_space.high)
print('Action space low: ', env.action_space.low)
print('Action space high: ', env.action_space.high)


env.close()
