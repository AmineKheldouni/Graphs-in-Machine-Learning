# -*- coding: utf-8 -*-

from monitor import *
import numpy as np

NAME = 'Walker2d'

if not os.path.exists('./results/'):
    os.mkdir('./results/')
if not os.path.exists('./results/'+NAME):
    os.mkdir('./results/'+NAME)

### BUILD GRAPH
def build_dics_walker2d(m, l, g, state, action):

    root_pos = state[0:2]
    root_vel = state[8:11]

    left_angle = state[2:5]
    right_angle = state[5:8]
    left_vel = state[11:14]
    right_vel = state[14:17]

    static_dict = {"globals": np.array([g], dtype=np.float32),
                  "nodes": np.array([[m] for i in range(7)], dtype=np.float32),
                  "edges": np.array([[l] for i in range(6)], dtype=np.float32),
                  "receivers": np.array([2, 3, 4, 5, 6, 6], dtype=np.float32),
                  "senders": np.array([0,1,2,3,4,5], dtype=np.float32)
            }


    dynamic_dict = {"globals": np.array([0.], dtype=np.float32),
                  "nodes": np.array([[root_pos[0], root_vel[1], root_pos[1], root_vel[2]],
                                     [l*right_angle[0], l*right_vel[0], right_angle[0], right_vel[0]],
                                     [l*right_angle[1], l*right_vel[1], right_angle[1], right_vel[1]],
                                     [l*right_angle[2], l*right_vel[2], right_angle[2], right_vel[2]],
                                     [l*left_angle[0], l*left_vel[0], left_angle[0], left_vel[0]],
                                     [l*left_angle[1], l*left_vel[1], left_angle[1], left_vel[1]],
                                     [l*left_angle[2], l*left_vel[2], left_angle[2], left_vel[2]]], dtype=np.float32),
                  "edges": np.array([[action[2]],
                                     [action[5]],
                                     [action[1]],
                                     [action[4]],
                                     [action[0]],
                                     [action[3]]], dtype=np.float32),
                   "receivers": np.array([2, 3, 4, 5, 6, 6], dtype=np.float32),
                   "senders": np.array([0,1,2,3,4,5], dtype=np.float32)
            }

    return static_dict, dynamic_dict


gn_monitor = GraphNetsMonitor(name = NAME,
                              env_name = 'Walker2d-v2',
                              build_dics_function = build_dics_walker2d,
                              nb_features_nodes = 4,
                              nb_features_edges = 1,
                              nb_features_globals = 1,
                              lr=1e-3,
                              max_iter = 1000,
                              nb_trajectories = 200)
gn_monitor.work()
