# -*- coding: utf-8 -*-

from monitor import *
import numpy as np

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


gn_monitor = GraphNetsMonitor(name = NAME,
                              env_name = 'HalfCheetah-v2',
                              build_dics_function = build_dics_halfcheetah,
                              nb_features = 4,
                              lr=1e-3,
                              max_iter = 1000,
                              nb_trajectories = 200)
gn_monitor.work()
