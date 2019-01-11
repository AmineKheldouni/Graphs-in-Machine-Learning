"""
Double DQN & Natural DQN comparison,
The Pendulum example.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: 1.0
gym: 0.8.0
"""


import gym
from RL_brain import DoubleDQN
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import tensorflow as tf
import time

env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(1)
MEMORY_SIZE = 5000
ACTION_SPACE = 11

sess = tf.Session()
with tf.variable_scope('Natural_DQN'):
    natural_DQN = DoubleDQN(
        n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=False, sess=sess
    )

with tf.variable_scope('Double_DQN'):
    double_DQN = DoubleDQN(
        n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=True, sess=sess, output_graph=True)

sess.run(tf.global_variables_initializer())


def train(RL):
    total_steps = 0
    observation = env.reset()
    while True:
        # if total_steps - MEMORY_SIZE > 8000: env.render()

        action = RL.choose_action(observation)

        f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)   # convert to [-2 ~ 2] float actions
        observation_, reward, done, info = env.step(np.array([f_action]))

        reward /= 10     # normalize to a range of (-1, 0). r = 0 when get upright
        # the Q target at upright state will be 0, because Q_target = r + gamma * Qmax(s', a') = 0 + gamma * 0
        # so when Q at this state is greater than 0, the agent overestimates the Q. Please refer to the final result.

        RL.store_transition(observation, action, reward, observation_)

        if total_steps > MEMORY_SIZE:   # learning
            RL.learn()

        if total_steps - MEMORY_SIZE > 20000:   # stop game
            break

        observation = observation_
        total_steps += 1
    return RL.q

# N = 20
# T = 500
# rewards = np.zeros((N,T))
# for i in range(N):
#     state = env.reset()
#     for step_index in range(T):
#         action = natural_DQN.choose_action(state)
#         f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)
#         state, reward, done, _ = env.step(np.array(f_action).reshape(-1))
#         rewards[i,step_index] = reward
#         if done:
#             print("Finished after iteration: ", step_index)
#             break
#
# rewards = np.mean(rewards, axis=0)

q_natural = train(natural_DQN)
# print('Rendering DQN trained agent')
# rewardsDQN = np.zeros((N,T))
# for i in range(N):
#     state = env.reset()
#     for step_index in range(T):
#         action = natural_DQN.choose_action(state)
#         f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)
#         env.render()
#         state, reward, done, _ = env.step(np.array(f_action).reshape(-1))
#         rewardsDQN[i,step_index] = reward
#         if done:
#             print("Finished after iteration: ", step_index)
#             break
#
# rewardsDQN = np.mean(rewardsDQN, axis=0)
# env.close()
#
# plt.plot(np.cumsum(rewardsDQN), color='green', label='DQN Agent trained')
# plt.plot(np.cumsum(rewards), color='red', label='Untrained Agent')
# plt.legend()
# plt.savefig('rewards_dqn.png')
# plt.show()
# plt.clf()

######################################################################
######################################################################
# rewards = np.zeros((N,T))
# for i in range(N):
#     state = env.reset()
#     for step_index in range(T):
#         action = double_DQN.choose_action(state)
#         f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)
#         state, reward, done, _ = env.step(np.array(f_action).reshape(-1))
#         rewards[i,step_index] = reward
#         if done:
#             print("Finished after iteration: ", step_index)
#             break
#
# rewards = np.mean(rewards, axis=0)
#
# q_double = train(double_DQN)
# print('Rendering DDQN trained agent')
# rewardsDQN = np.zeros((N,T))
# for i in range(N):
#     state = env.reset()
#     for step_index in range(T):
#         action = double_DQN.choose_action(state)
#         f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)
#         env.render()
#         state, reward, done, _ = env.step(np.array(f_action).reshape(-1))
#         rewardsDQN[i,step_index] = reward
#         if done:
#             print("Finished after iteration: ", step_index)
#             break
#
# rewardsDQN = np.mean(rewardsDQN, axis=0)
# env.close()
#
# plt.plot(np.cumsum(rewardsDQN), color='green', label='DQN Agent trained')
# plt.plot(np.cumsum(rewards), color='red', label='Untrained Agent')
# plt.legend()
# plt.savefig('rewards_ddqn.png')
# plt.show()
# plt.clf()


######################################################################
######################################################################


plt.plot(np.array(q_natural), c='green', label='DQN')
# plt.plot(np.array(q_double), c='b', label='DDQN')
plt.legend(loc='best')
plt.ylabel('Q eval')
plt.xlabel('training steps')
plt.grid()
plt.savefig('dqn_convergence.png')
plt.show()
plt.clf()
