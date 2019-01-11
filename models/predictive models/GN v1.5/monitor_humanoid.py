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

from model_v2 import *
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

class GraphNetsMonitor:
    def __init__(self,
                 name,
                 env_name,
                 build_dics_function,
                 nb_features_nodes = 2,
                 nb_features_edges = 2,
                 nb_features_globals = 2,
                 lr=1e-3,
                 max_iter = 1000,
                 nb_trajectories = 200,
                 m = 1.,
                 l = 1.,
                 g = -10,
                 plot_threshold = 200):

      self.name = name
      self.env_name = env_name
      self.build_dics_function = build_dics_function
      self.learning_rate = lr
      self.env = gym.make(env_name)
      self.max_iter = max_iter
      self.nb_trajectories = nb_trajectories
      self.sess = tf.Session()
      self.nb_features_nodes = nb_features_nodes
      self.nb_features_edges = nb_features_edges
      self.nb_features_globals = nb_features_globals

      self.m = m
      self.l = l
      self.g = g
      self.plot_threshold = plot_threshold

    def train(self):
      ### BUILD TRAINING AND TEST DATA, TRAIN MODEL
      dicts_in_static, dicts_in_dynamic, dicts_out_static, dicts_out_dynamic = sample_trajectories(self.env, self.nb_trajectories, self.m, self.l, self.g, self.build_dics_function)
      print("Trajectories created, building training data")
      X_train = utils_tf.concat([utils_tf.data_dicts_to_graphs_tuple(dicts_in_static), utils_tf.data_dicts_to_graphs_tuple(dicts_in_dynamic)], axis=1)
      Y_train = utils_tf.data_dicts_to_graphs_tuple(dicts_out_dynamic)

      print("Training data built, training the model")
      self.model = ForwardModel(name=self.name,
                                n_output_nodes=self.nb_features_nodes,
                                n_output_edges=self.nb_features_edges,
                                n_output_globals=self.nb_features_globals)

      output_train = self.model(X_train)
      self.loss = tf.reduce_mean((Y_train.nodes - output_train.nodes)**2) + tf.reduce_mean((Y_train.globals - output_train.globals)**2)
      optimizer = tf.train.AdamOptimizer(self.learning_rate)
      step_op = optimizer.minimize(self.loss)

      log_every_seconds = 4
      # Training
      try:
        self.sess.close()
      except NameError:
        pass
      self.sess = tf.Session()
      self.sess.run(tf.global_variables_initializer())
      losses_tr = []

      print("# (iteration number), T (elapsed seconds), "
            "Ltr (training 1-step loss), "
            "Lge4 (test/generalization rollout loss for 4-mass strings), "
            "Lge9 (test/generalization rollout loss for 9-mass strings)")

      start_time = time.time()
      last_log_time = start_time
      for iteration in range(0, self.max_iter):
        last_iteration = iteration
        train_values = self.sess.run({
            "step": step_op,
            "loss": self.loss,
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

      n_nodes = int(Y_train.nodes.shape[0]) // int(Y_train.n_node.shape[0])
      n_feats = int(Y_train.nodes.shape[1])
      n_globals = int(Y_train.globals.shape[1])
      print('n_nodes:', n_nodes)
      print('n_features:', n_feats)

      csv_columns = ['Node ' + str(i) + ' - Component ' + str(j) for i in range(n_nodes) for j in range(n_feats)] + ['Global '+str(i) for i in range(n_globals)]

      csv_train_gt = pd.DataFrame(columns=csv_columns)
      csv_train_pred = pd.DataFrame(columns=csv_columns)

      if not os.path.exists('./results/'+self.name+'/train'):
        os.mkdir('./results/'+self.name+'/train')

      coords_ref = self.sess.run(Y_train.globals)
      coords_output = self.sess.run(output_train.globals)

      for i in range(n_globals):
          coords_ref = self.sess.run(Y_train.globals)
          coords_output = self.sess.run(output_train.globals)

          csv_train_gt['Global ' + str(i)] = coords_ref[:,i]
          csv_train_pred['Global ' + str(i)] = coords_output[:,i]
          color = np.random.uniform(0, 1, 3)
          plt.plot(range(min(coords_ref.shape[0], self.plot_threshold)), coords_ref[:self.plot_threshold,i], color=tuple(color), label='Ground Truth Trajectory')
          plt.plot(range(min(coords_output.shape[0], self.plot_threshold)), coords_output[:self.plot_threshold,i], color=tuple(color), label='Predicted with GNN', linestyle='dashed')
          plt.legend()
          plt.savefig('./results/'+self.name+'/train/global ' + str(i) + '.png')
          # plt.show()
          plt.clf()

      for node in range(n_nodes):

          coords_ref = self.sess.run(Y_train.nodes)[node::n_nodes, :]
          coords_output = self.sess.run(output_train.nodes)[node::n_nodes, :]

          for i in range(n_feats):
              csv_train_gt['Node ' + str(node) + ' - Component ' + str(i)] = coords_ref[:,i]
              csv_train_pred['Node ' + str(node) + ' - Component ' + str(i)] = coords_output[:,i]
              color = np.random.uniform(0, 1, 3)
              plt.plot(range(min(coords_ref.shape[0], self.plot_threshold)), coords_ref[:self.plot_threshold,i], color=tuple(color), label='Ground Truth Trajectory')
              plt.plot(range(min(coords_output.shape[0], self.plot_threshold)), coords_output[:self.plot_threshold,i], color=tuple(color), label='Predicted with GNN', linestyle='dashed')
              plt.legend()
              plt.savefig('./results/'+self.name+'/train/component ' + str(i) + ', node ' + str(node) +'.png')
              # plt.show()
              plt.clf()


      csv_train_gt.to_csv('./results/'+self.name+'/train/GroundTruth_'+ str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '.csv', sep=',', index=False)
      csv_train_pred.to_csv('./results/'+self.name+'/train/Prediction_'+ str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '.csv', sep=',', index=False)

    def test(self):
        ### TEST RESULTS

        dicts_in_static_test, dicts_in_dynamic_test, dicts_out_static_test, dicts_out_dynamic_test = sample_trajectories(self.env, self.nb_trajectories, self.m, self.l, self.g, self.build_dics_function)

        print("Trajectories created, building testing data")

        X_test = utils_tf.concat([utils_tf.data_dicts_to_graphs_tuple(dicts_in_static_test), utils_tf.data_dicts_to_graphs_tuple(dicts_in_dynamic_test)], axis=1)
        Y_test = utils_tf.data_dicts_to_graphs_tuple(dicts_out_dynamic_test)
        n_globals = int(Y_test.globals.shape[1])

        print("Testing data built, testing the model")
        output_test = self.model(X_test)


        loss_test = tf.reduce_mean((Y_test.nodes - output_test.nodes)**2)

        test_values = self.sess.run({"input_graph": X_test, "target_graph": Y_test, "outputs": output_test, "loss": self.loss})

        ### Show trajectories for testing data

        n_nodes = int(Y_test.nodes.shape[0]) // int(Y_test.n_node.shape[0])
        n_feats = int(Y_test.nodes.shape[1])

        csv_columns = ['Node ' + str(i) + ' - Component ' + str(j) for i in range(n_nodes) for j in range(n_feats)] + ['Global '+str(i) for i in range(n_globals)]
        csv_test_gt = pd.DataFrame(columns=csv_columns)
        csv_test_pred = pd.DataFrame(columns=csv_columns)

        if not os.path.exists('./results/'+self.name+'/test'):
          os.mkdir('./results/'+self.name+'/test')

        coords_ref = self.sess.run(Y_test.globals)
        coords_output = self.sess.run(output_test.globals)

        for i in range(n_globals):

          csv_test_gt['Global ' + str(i)] = coords_ref[:,i]
          csv_test_pred['Global ' + str(i)] = coords_output[:,i]
          color = np.random.uniform(0, 1, 3)
          plt.plot(range(min(coords_ref.shape[0], self.plot_threshold)), coords_ref[:self.plot_threshold,i], color=tuple(color), label='Ground Truth Trajectory')
          plt.plot(range(min(coords_output.shape[0], self.plot_threshold)), coords_output[:self.plot_threshold,i], color=tuple(color), label='Predicted with GNN', linestyle='dashed')
          plt.legend()
          plt.savefig('./results/'+self.name+'/test/global ' + str(i) + '.png')
          # plt.show()
          plt.clf()

        for node in range(n_nodes):

            coords_ref = self.sess.run(Y_test.nodes)[node::n_nodes, :]
            coords_output = self.sess.run(output_test.nodes)[node::n_nodes, :]

            for i in range(n_feats):
                csv_test_gt['Node ' + str(node) + ' - Component ' + str(i)] = coords_ref[:,i]
                csv_test_pred['Node ' + str(node) + ' - Component ' + str(i)] = coords_output[:,i]
                color = np.random.uniform(0, 1, 3)
                plt.plot(range(min(coords_ref.shape[0], self.plot_threshold)), coords_ref[:self.plot_threshold,i], color=tuple(color), label='Ground Truth Trajectory')
                plt.plot(range(min(coords_output.shape[0], self.plot_threshold)), coords_output[:self.plot_threshold,i], color=tuple(color), label='Predicted with GNN', linestyle='dashed')
                plt.legend()
                plt.savefig('./results/'+self.name+'/test/component ' + str(i) + ', node ' + str(node) +'.png')
                # plt.show()
                plt.clf()

        csv_test_gt.to_csv('./results/'+self.name+'/test/GroundTruth_'+ str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '.csv', sep=',', index=False)
        csv_test_pred.to_csv('./results/'+self.name+'/test/Prediction_'+ str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '.csv', sep=',', index=False)

        self.env.close()
        try:
          self.sess.close()
        except err:
          pass

    def work(self):
      self.train()
      self.test()
