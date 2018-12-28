import os
import time
import gym
import numpy as np
import matplotlib.pyplot as plt
import keras.utils as np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, UpSampling2D, Activation
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
from keras import backend as K
K.tensorflow_backend._get_available_gpus()


env = gym.make('Pendulum-v0')
# env = gym.make('Acrobot-v1')
# env = gym.make('HalfCheetah-v2')
env.reset()
# print("Bounds for observation space:")
# print(env.observation_space.low, env.observation_space.high)
# print("Bounds for action space:")
# print(env.action_space.low, env.action_space.high)

def sample_trajectories(env, N):
    X = []
    Y = []
    state = env.reset()
    while (len(X) < N):
      for i in range(N):
          action = env.action_space.sample()
          next_state, reward, done, info = env.step(action)
          example = [obs for obs in state]
          example.append(action)
          X.append(np.array(example))
          Y.append(next_state)
          state = next_state
          if done or len(X) == N:
              break
    return X,Y

X_train, Y_train = sample_trajectories(env, 10000)
X_test, Y_test = sample_trajectories(env, 1000)
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
print(X_test.shape)
print(Y_test.shape)

def linear_nn(X, Y):
    model = Sequential()
    model.add(Dense(20, input_dim=X.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(Y.shape[1], kernel_initializer='normal'))

    model.compile(loss='mse', optimizer='adam')
    hist = model.fit(X, Y, validation_split=0.05, epochs=30, batch_size=64, verbose=1, shuffle=True)

    return hist, model

regression_history, regression_model = linear_nn(X_train, Y_train)

plt.figure(figsize=(8,8))
plt.plot(regression_history.history['loss'])
plt.plot(regression_history.history['val_loss'])
plt.title('CNN regression loss with Adam optimizer')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#
test_loss =  regression_model.evaluate(X_test, Y_test)
print("Cross-entropy loss for test dataset: ",test_loss)


current_state = env.reset()
gt_path = []
predicted_path = []
for step_index in range(1000):
    # time.sleep(0.05)
    # env.render()
    action = env.action_space.sample()

    if isinstance(current_state, np.ndarray):
        x = [s for s in current_state]
    else:
        x = [current_state]
    if isinstance(action, np.ndarray):
        x += [a for a in action]
    else:
        x += [action]

    y_pred = regression_model.predict(np.array(x).reshape((1,-1)))

    next_state, reward, done, info = env.step(action)
    gt_path.append(next_state)
    predicted_path.append(y_pred)
    current_state = next_state
    if done:
        print("Finished after iteration: ", step_index)
        break

gt_path = np.array(gt_path)
predicted_path = np.array(predicted_path).reshape(gt_path.shape)
fig, ax = plt.subplots(len(current_state),1)
print("predicted_path: ", predicted_path.shape)
print("gt_path: ", gt_path.shape)
for i in range(len(current_state)):
    color = np.random.uniform(0, 1, 3)
    ax[i].plot(range(gt_path.shape[0]), gt_path.T[i], color=tuple(color), label='Ground Truth Trajectory')
    ax[i].plot(range(predicted_path.shape[0]), predicted_path.T[i], color=tuple(color), label='Predicted with Linear NN', linestyle='dashed')
plt.legend()
plt.draw()
plt.show()
# scores = []
# choices = []
# for each_game in range(100):
#     score = 0
#     prev_obs = []
#     for step_index in range(goal_steps):
#
#         # time.sleep(0.05)
#         # env.render()
#         if len(prev_obs)==0:
#             action = env.action_space.sample()
#         else:
#             action = np.argmax(trained_model.predict(prev_obs.reshape(-1, len(prev_obs)))[0])
#
#         choices.append(action)
#         new_observation, reward, done, info = env.step(action)
#         prev_obs = new_observation
#         score+=reward
#         if done:
#             print("Finished after iteration: ", step_index)
#             break
#
#     env.reset()
#     scores.append(score)
# env.close()
# print(scores)
# print('Average Score:', sum(scores)/len(scores))
# print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
#
# score = 0
# prev_obs = []
# for step_index in range(goal_steps):
#     time.sleep(0.05)
#     env.render()
#     if len(prev_obs)==0:
#         action = env.action_space.sample()
#     else:
#         action = np.argmax(trained_model.predict(prev_obs.reshape(-1, len(prev_obs)))[0])
#
#     new_observation, reward, done, info = env.step(action)
#     prev_obs = new_observation
#     score+=reward
#     if done:
#         print("Finished after iteration: ", step_index)
#         break
# print("Score: ", score)
# env.reset()
