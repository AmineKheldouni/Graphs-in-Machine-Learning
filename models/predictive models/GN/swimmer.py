# -*- coding: utf-8 -*-

from monitor import *
import numpy as np

NAME = 'Swimmer'

if not os.path.exists('./results/'):
    os.mkdir('./results/')
if not os.path.exists('./results/'+NAME):
    os.mkdir('./results/'+NAME)

### BUILD GRAPH


def build_dics_swimmer(m, l, g, state, action):

    positions = state[:4]
    velocities = state[4:]

    static_dict = {"globals": np.array([g], dtype=np.float32),
                  "nodes": np.array([[0], [0], [0], [0]], dtype=np.float32),
                  "edges": np.array([[l], [l], [l]], dtype=np.float32),
                  "receivers": np.array([0, 1, 2], dtype=np.float32),
                  "senders": np.array([1, 2, 3], dtype=np.float32)
            }


    dynamic_dict = {"globals": np.array([0.], dtype=np.float32),
                  "nodes": np.array([[positions[i], velocities[i]] for i in range(4)], dtype=np.float32),
                  "edges": np.array([[action[0]], [0], [action[1]]], dtype=np.float32),
                  "receivers": np.array([0, 1, 2], dtype=np.float32),
                  "senders": np.array([1, 2, 3], dtype=np.float32)
            }

    return static_dict, dynamic_dict



gn_monitor = GraphNetsMonitor(name = NAME,
                              env_name = 'Swimmer-v2',
                              build_dics_function = build_dics_swimmer,
                              nb_features = 2,
                              lr=1e-3,
                              max_iter = 1000,
                              nb_trajectories = 200)
gn_monitor.work()
