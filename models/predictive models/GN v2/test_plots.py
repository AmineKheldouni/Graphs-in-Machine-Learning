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

name = "Walker2d"
time_save = "2019-01-07_19_22_50"
file_path_prediction = "./results/" + name + "/test/old/GroundTruth_" + time_save + ".csv"
file_path_groundtruth = "./results/" + name + "/test/old/Prediction_" + time_save + ".csv"

#csv_columns = ['Node ' + str(i) + ' - Component ' + str(j) for i in range(n_nodes) for j in range(n_feats)]
csv_test_gt = pd.read_csv(file_path_groundtruth)
csv_test_pred = pd.read_csv(file_path_prediction)

plot_threshold = 200
n_nodes = 7
n_feats = 4

for node in range(n_nodes):

    for i in range(n_feats):
        coords_ref = csv_test_gt['Node ' + str(node) + ' - Component ' + str(i)]
        coords_output = csv_test_pred['Node ' + str(node) + ' - Component ' + str(i)]
        color = np.random.uniform(0, 1, 3)
        plt.plot(range(min(coords_ref.shape[0], plot_threshold)), coords_ref[:plot_threshold], color=tuple(color), label='Ground Truth Trajectory')
        plt.plot(range(min(coords_output.shape[0], plot_threshold)), coords_output[:plot_threshold], color=tuple(color), label='Predicted with GNN', linestyle='dashed')
        plt.title('Test Trajectory - Ground Truth vs Prediction - component ' + str(i) + ', node ' + str(node))
        plt.legend()
        plt.savefig('./results/'+name+'/test/component ' + str(i) + ', node ' + str(node) +'.png')
        plt.show()
        plt.clf()

csv_test_gt.to_csv('./results/'+name+'/test/GroundTruth_'+ time_save + '.csv', sep=',', index=False)
csv_test_pred.to_csv('./results/'+name+'/test/Prediction_'+ time_save + '.csv', sep=',', index=False)