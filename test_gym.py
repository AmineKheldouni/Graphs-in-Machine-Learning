import argparse
import sys
import gym
import time
from gym import wrappers, logger

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


env = gym.make('Pendulum-v0')

env.seed(0)
env.mode = 'human'

agent = RandomAgent(env.action_space)

observation = env.reset()
for t in range(500):
    time.sleep(0.05)
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample())
    if done:
        print("Finished after {} timesteps".format(t+1))
        break
env.close()
