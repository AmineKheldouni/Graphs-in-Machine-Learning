# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 12:35:23 2018

@author: ocliv
"""
import os
from graph_nets import blocks
from graph_nets import utils_tf
from graph_nets.demos import models
from graph_nets import modules
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
import tensorflow as tf
import sonnet as snt

try:
  import seaborn as sns
except ImportError:
  pass
else:
  sns.reset_orig()

SEED = 1
np.random.seed(SEED)
tf.set_random_seed(SEED)


### BUILD GRAPH

if True:

    def build_dics_cartpole(m, l, g, cart_pos, cart_speed, pole_angle, pole_speed, action):


        static_dict = {"globals": np.array([g], dtype=np.float32),
                      "nodes": np.array([[m], [0]], dtype=np.float32),
                      "edges": np.array([[l]], dtype=np.float32),
                      "receivers": np.array([0]),
                      "senders": np.array([1])
                }


        dynamic_dict = {"globals": np.array([0.], dtype=np.float32),
                      "nodes": np.array([[pole_angle, pole_speed], [cart_pos, cart_speed]], dtype=np.float32),
                      "edges": np.array([[action]], dtype=np.float32),
                      "receivers": np.array([0]),
                      "senders": np.array([1])
                }

        return static_dict, dynamic_dict



    m = 1.
    l = 1.
    g = -10.



    ### CREATE MODEL CLASS


    def make_mlp_model(n_latent = 128, n_hidden_layers = 2, n_output = 2):

      return snt.Sequential([
          snt.nets.MLP([n_latent] * n_hidden_layers, activate_final=True),
          snt.nets.MLP([n_output], activate_final = False)
      ])

    class MLPGraphNetwork(snt.AbstractModule):
      def __init__(self, name="MLPGraphNetwork"):
        super(MLPGraphNetwork, self).__init__(name=name)
        with self._enter_variable_scope():
          self._network = modules.GraphNetwork(make_mlp_model, make_mlp_model,
                                               make_mlp_model)

      def _build(self, inputs):
        return self._network(inputs)


    class ForwardModel(snt.AbstractModule):

        def __init__(self, n_latent = 2, name = "CartPole"):
            super(ForwardModel, self).__init__(name=name)
            self.n_latent = n_latent
            self.n_output = 2

            with self._enter_variable_scope():
                self._network_1 = MLPGraphNetwork()
                self._network_2 = MLPGraphNetwork()

        def _build(self, graph_input):
            graph_latent = self._network_1(graph_input)
            graph_concat = utils_tf.concat([graph_input, graph_latent], axis=1)
            output = self._network_2(graph_concat)
            return output

    ### BUILD TRAINING AND TEST DATA, TRAIN MODEL

    env = gym.make('CartPole-v0')
    env.reset()

    def sample_trajectories(env, N, m, l, g):
        dicts_in_static = []
        dicts_in_dynamic = []
        dicts_out_static = []
        dicts_out_dynamic = []
        cart_pos, cart_speed, pole_angle, pole_speed = env.reset()
        while (len(dicts_in_static) < N):
          for i in range(N):

              action = env.action_space.sample()
              [cart_pos_next, cart_speed_next, pole_angle_next, pole_speed_next], _, done, _ = env.step(action)

              dict_in_static, dict_in_dynamic = build_dics_cartpole(m, l, g, cart_pos, cart_speed, pole_angle, pole_speed, action)
              dict_out_static, dict_out_dynamic = build_dics_cartpole(m, l, g, cart_pos_next, cart_speed_next, pole_angle_next, pole_speed_next, action)

              dicts_in_static.append( dict_in_static )
              dicts_in_dynamic.append( dict_in_dynamic )
              dicts_out_static.append( dict_out_static )
              dicts_out_dynamic.append( dict_out_dynamic )

              cart_pos, cart_speed, pole_angle, pole_speed = cart_pos_next, cart_speed_next, pole_angle_next, pole_speed_next

              if done or len(dicts_in_static) == N:
                  break
        return dicts_in_static, dicts_in_dynamic, dicts_out_static, dicts_out_dynamic

    dicts_in_static, dicts_in_dynamic, dicts_out_static, dicts_out_dynamic = sample_trajectories(env, 2000, m, l, g)

    print("Trajectories created, building training data")

    X_train = utils_tf.concat([utils_tf.data_dicts_to_graphs_tuple(dicts_in_static), utils_tf.data_dicts_to_graphs_tuple(dicts_in_dynamic)], axis=1)
    Y_train = utils_tf.data_dicts_to_graphs_tuple(dicts_out_dynamic)

    print("Training data built, training the model")
    model = ForwardModel()
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
    for iteration in range(0, 10000):
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

coords_output = sess.run(Y_train.nodes)[1:301:2, :]
coords_ref = sess.run(output_train.nodes)[1:301:2, :]

for i in range(2):
    color = np.random.uniform(0, 1, 3)
    plt.plot(range(coords_ref.shape[0]), coords_ref[:,i], color=tuple(color), label='Ground Truth Trajectory')
    plt.plot(range(coords_output.shape[0]), coords_output[:,i], color=tuple(color), label='Predicted with Linear NN', linestyle='dashed')
    plt.title('Train - Ground Truth trajectory vs Prediction '+ str(i)+'-th component of state vector')
    plt.legend()
    plt.draw()
    plt.show()


### TEST RESULTS

dicts_in_static_test, dicts_in_dynamic_test, dicts_out_static_test, dicts_out_dynamic_test = sample_trajectories(env, 2000, m, l, g)

print("Trajectories created, building testing data")

X_test = utils_tf.concat([utils_tf.data_dicts_to_graphs_tuple(dicts_in_static_test), utils_tf.data_dicts_to_graphs_tuple(dicts_in_dynamic_test)], axis=1)
Y_test = utils_tf.data_dicts_to_graphs_tuple(dicts_out_dynamic_test)

print("Testing data built, testing the model")
output_test = model(X_test)


loss_test = tf.reduce_mean((Y_test.nodes - output_test.nodes)**2)

test_values = sess.run({"input_graph": X_test, "target_graph": Y_test, "outputs": output_test, "loss": loss})

coords_output = sess.run(Y_test.nodes)[1:301:2, :]
coords_ref = sess.run(output_test.nodes)[1:301:2, :]

for i in range(2):
    color = np.random.uniform(0, 1, 3)
    plt.plot(range(coords_ref.shape[0]), coords_ref[:,i], color=tuple(color), label='Ground Truth Trajectory')
    plt.plot(range(coords_output.shape[0]), coords_output[:,i], color=tuple(color), label='Predicted with Linear NN', linestyle='dashed')
    plt.title('Test - Ground Truth trajectory vs Prediction '+ str(i)+'-th component of state vector')
    plt.legend()
    plt.draw()
    plt.show()
