import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time
# Credit: https://github.com/udacity/deep-reinforcement-learning

env = gym.make('LunarLander-v2')
env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Action shape: ', env.action_space.shape)
# print('Number of actions: ', env.action_space.n)

from DQNAgent import Agent

agent = Agent(state_size=env.observation_space.shape[0], action_size=1, seed=0)
T = 500

# watch an untrained agent
state = env.reset()
rewards = []
for step_index in range(T):
    action = agent.act(state)
    env.render()
    time.sleep(0.01)
    state, reward, done, _ = env.step(action)
    rewards.append(reward)
    if done:
        print("Finished after iteration: ", step_index)
        break

env.close()
plt.plot(np.cumsum(rewards))
plt.show()

def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            break
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint' + env_name + '.pth')
    return scores

scores = dqn()

# plot the scores
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(np.arange(len(scores)), scores)
# plt.ylabel('Score')
# plt.xlabel('Episode #')
# plt.show()


# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('checkpoint' + env_name + '.pth'))
N = 3
rewardsDQN = np.zeros((N,T))
for i in range(N):
    state = env.reset()
    rewards = []
    for step_index in range(T):
        action = agent.act(state)
        env.render()
        time.sleep(0.01)
        state, reward, done, _ = env.step(action)
        rewardsDQN[i,step_index] = reward
        if done:
            print("Finished after iteration: ", step_index)
            break

rewardsDQN = np.mean(rewardsDQN, axis=0)
env.close()

plt.plot(np.cumsum(rewardsDQN), color='green', label='DQN Agent trained')
plt.plot(np.cumsum(rewards), color='red', label='Untrained Agent')
plt.show()
