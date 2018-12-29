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

def build_dics_pendulum(m, l, g, theta_0, theta_dot_0, torque_0):
    
    
    static_dict = {"globals": [g],
                  "nodes": [None, m],
                  "edges": [l],
                  "receivers": [0],
                  "senders": [1]
            }
    
    
    dynamic_dict = {"globals": [None],
                  "nodes": [None, [theta_0, theta_dot_0]],
                  "edges": [torque_0],
                  "receivers": [0],
                  "senders": [1]
            }
    
    return static_dict, dynamic_dict

        

m = 1.
l = 1.
g = -10.


static_dict, dynamic_dict = build_dics_pendulum(m, l, g, theta_0, theta_dot_0, torque_0)


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
    
    def __init__(self, n_latent = 2, name = "Forward model Pendulum-v0 first version"):
        super(ForwardModel, self).__init__(name=name)
        self.n_latent = n_latent
        self.n_output = 2
        
        with self._enter_variable_scope():
            self._network_1 = MLPGraphNetwork()
            self._network_2 = MLPGraphNetwork()
        
    def _build(self, graph_input):
        graph_latent = self._network_1(graph_input)
        graph_concat = utils_tf.concat([graph_input, graph_latent])
        output = self._network_2(graph_concat)
        return output

### BUILD TRAINING AND TEST DATA, TRAIN MODEL

env = gym.make('Pendulum-v0')
env.reset()

def sample_trajectories(env, N, m, l, g):
    dicts_in = []
    dicts_out = []
    state = env.reset()
    while (len(X) < N):
      for i in range(N):
          action = env.action_space.sample()
          theta_prev, thetadot_prev = env.state
          _, _, done, _ = env.step(action)
          theta_next, thetadot_next = env.state
          dicts_in.append( build_dics_pendulum(m, l, g, theta_prev, thetadot_prev, action) )
          dicts_out.append( build_dics_pendulum(m, l, g, theta_next, thetadot_next, action) )
          if done or len(X) == N:
              break
    return dicts_in, dicts_out
 


 # To be continued

### TEST


