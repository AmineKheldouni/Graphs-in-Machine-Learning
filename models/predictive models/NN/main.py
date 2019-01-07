import os
import time
import gym
import numpy as np
import matplotlib.pyplot as plt
import keras.utils as np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from keras.optimizers import SGD, Adam
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
import pandas as pd
import datetime

# print("Bounds for observation space:")
# print(env.observation_space.low, env.observation_space.high)
# print("Bounds for action space:")
# print(env.action_space.low, env.action_space.high)
NAME = 'CartPole'

if not os.path.exists('./results/'):
    os.mkdir('./results/')
if not os.path.exists('./results/'+NAME):
    os.mkdir('./results/'+NAME)

env = gym.make('CartPole-v0')

def sample_trajectories(env, N):
    X = []
    Y = []
    while (len(X) < N):
      state = env.reset()
      for i in range(100):
          action = env.action_space.sample()
          next_state, reward, done, info = env.step(action)
          if isinstance(state, np.ndarray):
              example = [s for s in state]
          else:
              example = [state]
          if isinstance(action, np.ndarray):
              example += [a for a in action]
          else:
              example += [action]

          X.append(np.array(example))
          Y.append(next_state)
          state = next_state
          if done or len(X) >= N:
              break
    return X,Y

X_train, Y_train = sample_trajectories(env, 20000)
X_test, Y_test = sample_trajectories(env, 2000)
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test.shape)

def linear_nn(X, Y):
    model = Sequential()
    model.add(Dense(128, input_dim=X.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(Y.shape[1], kernel_initializer='normal'))

    model.compile(loss='mse', optimizer='adam')
    hist = model.fit(X, Y, validation_split=0.1, epochs=200, batch_size=128, verbose=1, shuffle=True)

    return hist, model

regression_history, regression_model = linear_nn(X_train, Y_train)


if not os.path.exists('./results/'+NAME+'/train'):
  os.mkdir('./results/'+NAME+'/train')

csv_columns = ['Component ' + str(j) for j in range(env.observation_space.shape[0])]

csv_train_gt = pd.DataFrame(columns=csv_columns)
csv_train_pred = pd.DataFrame(columns=csv_columns)

Y_train_pred = regression_model.predict(X_train)
for j in range(env.observation_space.shape[0]):
    csv_train_gt['Component ' + str(j)] = Y_train[:,j]
    csv_train_pred['Component ' + str(j)] = Y_train_pred[:,j]
    color = np.random.uniform(0, 1, 3)
    plt.plot(range(100), Y_train.T[j][:100], color=tuple(color), label='Ground Truth Trajectory')
    plt.plot(range(100), Y_train_pred.T[j][:100], color=tuple(color), label='Predicted with Linear NN', linestyle='dashed')
    plt.title('Ground Truth trajectory vs Prediction '+ str(j)+'-th component of state vector')
    plt.legend()
    plt.savefig('./results/' + NAME + '/train/' + str(j) + '-th component of state vector.png')
    plt.clf()
    # plt.show()

csv_train_gt.to_csv('./results/'+NAME+'/train/GroundTruth_'+ str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")) + '.csv', sep=',', index=False)
csv_train_pred.to_csv('./results/'+NAME+'/train/Prediction_'+ str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")) + '.csv', sep=',', index=False)

current_state = env.reset()
Y_test = []
Y_test_pred = []
env.reset()
for step_index in range(100):
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
    Y_test.append(next_state)
    Y_test_pred.append(y_pred)
    current_state = next_state
    if done:
        print("Finished after iteration: ", step_index)
        break

Y_test = np.array(Y_test)
Y_test_pred = np.array(Y_test_pred).reshape(Y_test.shape)


if not os.path.exists('./results/'+NAME+'/test'):
  os.mkdir('./results/'+NAME+'/test')

csv_test_gt = pd.DataFrame(columns=csv_columns)
csv_test_pred = pd.DataFrame(columns=csv_columns)

for j in range(env.observation_space.shape[0]):
    csv_test_gt['Component ' + str(j)] = Y_test[:,j]
    csv_test_pred['Component ' + str(j)] = Y_test_pred[:,j]

csv_test_gt.to_csv('./results/'+NAME+'/test/GroundTruth_'+ str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")) + '.csv', sep=',', index=False)
csv_test_pred.to_csv('./results/'+NAME+'/test/Prediction_'+ str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")) + '.csv', sep=',', index=False)


print("Y_test_pred: ", Y_test_pred.shape)
print("Y_test: ", Y_test.shape)
for i in range(len(current_state)):
    color = np.random.uniform(0, 1, 3)
    plt.plot(range(Y_test.shape[0]), Y_test.T[i], color=tuple(color), label='Ground Truth Trajectory')
    plt.plot(range(Y_test_pred.shape[0]), Y_test_pred.T[i], color=tuple(color), label='Predicted with Linear NN', linestyle='dashed')
    plt.title('Ground Truth trajectory vs Prediction '+ str(i)+'-th component of state vector')
    plt.legend()
    plt.savefig('./results/' + NAME + '/test/' + str(i) + '-th component of state vector.png')
    plt.clf()
    # plt.show()




### LOSS PLOTS ###
# plt.figure(figsize=(8,8))
# plt.plot(regression_history.history['loss'])
# plt.plot(regression_history.history['val_loss'])
# plt.title('CNN regression loss with Adam optimizer')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
#
# test_loss =  regression_model.evaluate(X_test, Y_test)
# print("Cross-entropy loss for test dataset: ",test_loss)
#
#
