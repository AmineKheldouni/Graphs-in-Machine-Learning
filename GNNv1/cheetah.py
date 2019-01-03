# -*- coding: utf-8 -*-
import os
import sys

import datetime
import time

import gym
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
import tensorflow as tf
import pandas as pd

from model import *
from sampler import *
try:
  import seaborn as sns
except ImportError:
  pass
else:
  sns.reset_orig()

SEED = 1
np.random.seed(SEED)
tf.set_random_seed(SEED)

nb_nodes = 7
m = 1.
l = 1.
g = -10.
FEATURES_NUMBER = 4
TRAJECTORIES_NUMBER = 200
MAX_ITER = 1500
NAME = 'HalfCheetah'

if not os.path.exists('./results/'):
    os.mkdir('./results/')
if not os.path.exists('./results/'+NAME):
    os.mkdir('./results/'+NAME)

### BUILD GRAPH
def build_dics_halfcheetah(m, l, g, state, action):

    root_pos = state[0:2]
    root_vel = state[8:11]

    bot_angle = state[2:5]
    front_angle = state[5:8]
    bot_vel = state[11:14]
    front_vel = state[14:17]
    bot_action = action[0:3]
    front_action = action[3:]

    static_dict = {"globals": np.array([g], dtype=np.float32),
                  "nodes": np.array([[m] for i in range(7)], dtype=np.float32),
                  "edges": np.array([[l] for i in range(6)], dtype=np.float32),
                  "receivers": np.array([2,1,0,0,4,5], dtype=np.float32),
                  "senders": np.array([3,2,1,4,5,6], dtype=np.float32)
            }


    dynamic_dict = {"globals": np.array([0.], dtype=np.float32),
                  "nodes": np.array([[root_pos[0], root_vel[1], root_pos[1], root_vel[2]],
                                     [l*front_angle[0], l*front_vel[0], front_angle[0], front_vel[0]],
                                     [l*front_angle[1], l*front_vel[1], front_angle[1], front_vel[1]],
                                     [l*front_angle[2], l*front_vel[2], front_angle[2], front_vel[2]],
                                     [l*bot_angle[0], l*bot_vel[0], bot_angle[0], bot_vel[0]],
                                     [l*bot_angle[1], l*bot_vel[1], bot_angle[1], bot_vel[1]],
                                     [l*bot_angle[2], l*bot_vel[2], bot_angle[2], bot_vel[2]]], dtype=np.float32),
                  "edges": np.array([[front_action[2]],
                                     [front_action[1]],
                                     [front_action[0]],
                                     [bot_action[0]],
                                     [bot_action[1]],
                                     [bot_action[2]]], dtype=np.float32),
                  "receivers": np.array([2,1,0,0,4,5], dtype=np.float32),
                  "senders": np.array([3,2,1,4,5,6], dtype=np.float32)
            }

    return static_dict, dynamic_dict


### BUILD TRAINING AND TEST DATA, TRAIN MODEL

env = gym.make('HalfCheetah-v2')

dicts_in_static, dicts_in_dynamic, dicts_out_static, dicts_out_dynamic = sample_trajectories(env, TRAJECTORIES_NUMBER, m, l, g, build_dics_halfcheetah)

print("Trajectories created, building training data")
X_train = utils_tf.concat([utils_tf.data_dicts_to_graphs_tuple(dicts_in_static), utils_tf.data_dicts_to_graphs_tuple(dicts_in_dynamic)], axis=1)
Y_train = utils_tf.data_dicts_to_graphs_tuple(dicts_out_dynamic)

print("Training data built, training the model")
model = ForwardModel(name=NAME, n_output_final=FEATURES_NUMBER)
output_train = model(X_train)

loss = tf.reduce_mean((Y_train.nodes - output_train.nodes)**2)

learning_rate = 1e-3
optimizer = tf.train.AdamOptimizer(learning_rate)
step_op = optimizer.minimize(loss)


log_every_seconds = 20

# Training

try:
  sess.close()
except NameError:
  pass
sess = tf.Session()
sess.run(tf.global_variables_initializer())

losses_tr = []

print("# (iteration number), T (elapsed seconds), "
      "Ltr (training 1-step loss), "
      "Lge4 (test/generalization rollout loss for 4-mass strings), "
      "Lge9 (test/generalization rollout loss for 9-mass strings)")

start_time = time.time()
last_log_time = start_time
for iteration in range(0, MAX_ITER):
  last_iteration = iteration
  train_values = sess.run({
      "step": step_op,
      "loss": loss,
      "input_graph": X_train,
      "target_nodes": Y_train.nodes,
      "outputs": output_train
  })
  the_time = time.time()
  elapsed_since_last_log = the_time - last_log_time
  if elapsed_since_last_log > log_every_seconds:
    last_log_time = the_time
    elapsed = time.time() - start_time
    losses_tr.append(train_values["loss"])
    print("# {:05d}, T {:.1f}, Ltr {:.4f}".format(
        iteration, elapsed, train_values["loss"]))

### TRAIN RESULTS

### Show trajectories for training data

n_nodes = int(Y_train.nodes.shape[0]) // int(Y_train.n_node.shape[0])
n_feats = int(Y_train.nodes.shape[1])
print('n_nodes:', n_nodes)
print('n_features:', n_feats)

csv_columns = ['Node ' + str(i) + ' - Component ' + str(j) for i in range(n_nodes) for j in range(n_feats)]

csv_train_gt = pd.DataFrame(columns=csv_columns)
csv_train_pred = pd.DataFrame(columns=csv_columns)
for node in range(n_nodes):

    coords_ref = sess.run(Y_train.nodes)[node::n_nodes, :]
    coords_output = sess.run(output_train.nodes)[node::n_nodes, :]

    for i in range(n_feats):
        csv_train_gt['Node ' + str(node) + ' - Component ' + str(i)] = coords_ref[:,i]
        csv_train_pred['Node ' + str(node) + ' - Component ' + str(i)] = coords_output[:,i]
        color = np.random.uniform(0, 1, 3)
        plt.plot(range(coords_ref.shape[0]), coords_ref[:,i], color=tuple(color), label='Ground Truth Trajectory')
        plt.plot(range(coords_output.shape[0]), coords_output[:,i], color=tuple(color), label='Predicted with GNN', linestyle='dashed')
        plt.title('Train Trajectory - Ground Truth vs Prediction - component ' + str(i) + ', node ' + str(node))
        plt.legend()
        plt.draw()
        plt.show()


if not os.path.exists('./results/'+NAME+'/train'):
  os.mkdir('./results/'+NAME+'/train')
csv_train_gt.to_csv('./results/'+NAME+'/train/GroundTruth_'+ str(datetime.datetime.now()) + '.csv', sep=',', index=False)
csv_train_pred.to_csv('./results/'+NAME+'/train/Prediction_'+ str(datetime.datetime.now()) + '.csv', sep=',', index=False)


### TEST RESULTS

dicts_in_static_test, dicts_in_dynamic_test, dicts_out_static_test, dicts_out_dynamic_test = sample_trajectories(env, TRAJECTORIES_NUMBER, m, l, g, build_dics_halfcheetah)

print("Trajectories created, building testing data")

X_test = utils_tf.concat([utils_tf.data_dicts_to_graphs_tuple(dicts_in_static_test), utils_tf.data_dicts_to_graphs_tuple(dicts_in_dynamic_test)], axis=1)
Y_test = utils_tf.data_dicts_to_graphs_tuple(dicts_out_dynamic_test)

print("Testing data built, testing the model")
output_test = model(X_test)


loss_test = tf.reduce_mean((Y_test.nodes - output_test.nodes)**2)

test_values = sess.run({"input_graph": X_test, "target_graph": Y_test, "outputs": output_test, "loss": loss})

### Show trajectories for testing data

n_nodes = int(Y_test.nodes.shape[0]) // int(Y_test.n_node.shape[0])
n_feats = int(Y_test.nodes.shape[1])

csv_columns = ['Node ' + str(i) + ' - Component ' + str(j) for i in range(n_nodes) for j in range(n_feats)]
csv_test_gt = pd.DataFrame(columns=csv_columns)
csv_test_pred = pd.DataFrame(columns=csv_columns)
for node in range(n_nodes):

    coords_output = sess.run(Y_test.nodes)[node::n_nodes, :]
    coords_ref = sess.run(output_test.nodes)[node::n_nodes, :]

    for i in range(n_feats):
        csv_test_gt['Node ' + str(node) + ' - Component ' + str(i)] = coords_ref[:,i]
        csv_test_pred['Node ' + str(node) + ' - Component ' + str(i)] = coords_output[:,i]
        color = np.random.uniform(0, 1, 3)
        plt.plot(range(coords_ref.shape[0]), coords_ref[:,i], color=tuple(color), label='Ground Truth Trajectory')
        plt.plot(range(coords_output.shape[0]), coords_output[:,i], color=tuple(color), label='Predicted with GNN', linestyle='dashed')
        plt.title('Test Trajectory - Ground Truth vs Prediction - component ' + str(i) + ', node ' + str(node))
        plt.legend()
        plt.draw()
        plt.show()


if not os.path.exists('./results/'+NAME+'/test'):
  os.mkdir('./results/'+NAME+'/test')
csv_test_gt.to_csv('./results/'+NAME+'/test/GroundTruth_'+ str(datetime.datetime.now()) + '.csv', sep=',', index=False)
csv_test_pred.to_csv('./results/'+NAME+'/test/Prediction_'+ str(datetime.datetime.now()) + '.csv', sep=',', index=False)

env.close()
sess.close()
