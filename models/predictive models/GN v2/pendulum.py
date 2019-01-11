# -*- coding: utf-8 -*-

from monitor_v2 import *
import numpy as np

NAME = 'Pendulum'

if not os.path.exists('./results/'):
    os.mkdir('./results/')
if not os.path.exists('./results/'+NAME):
    os.mkdir('./results/'+NAME)

### BUILD GRAPH
def build_dics_pendulum(m, l, g, state, action):

    theta = np.arctan2(state[1], state[0])
    theta_dot = state[2]
    action = action.sum()
    static_dict = {"globals": np.array([g], dtype=np.float32),
                  "nodes": np.array([[0.], [m]], dtype=np.float32),
                  "edges": np.array([[l]], dtype=np.float32),
                  "receivers": np.array([1]),
                  "senders": np.array([0])
            }


    dynamic_dict = {"globals": np.array([0.], dtype=np.float32),
                  "nodes": np.array([[0., 0.], [theta, theta_dot]], dtype=np.float32),
                  "edges": np.array([[action]], dtype=np.float32),
                  "receivers": np.array([1]),
                  "senders": np.array([0])
            }

    return static_dict, dynamic_dict


gn_monitor = GraphNetsMonitor(name = NAME,
                              env_name = 'Pendulum-v0',
                              build_dics_function = build_dics_pendulum,
                              nb_features_nodes = 2,
                              nb_features_edges = 1,
                              nb_features_globals = 1,
                              lr=1e-3,
                              max_iter = 500,
                              nb_trajectories = 500,
                              T = 5)
gn_monitor.work()
