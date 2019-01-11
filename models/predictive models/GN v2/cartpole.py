# -*- coding: utf-8 -*-

from monitor import *
import numpy as np
NAME = 'CartPole'

if not os.path.exists('./results/'):
    os.mkdir('./results/')
if not os.path.exists('./results/'+NAME):
    os.mkdir('./results/'+NAME)

def build_dics_cartpole(m, l, g, state, action):

    cart_pos = state[0]
    cart_vel = state[1]
    pole_angle = state[2]
    pole_angvel = state[3]
    pole_pos = l * pole_angle
    pole_vel = l * pole_angvel
    cart_angle = cart_pos/l
    cart_angvel = cart_vel/l

    static_dict = {"globals": np.array([g], dtype=np.float32),
                  "nodes": np.array([[m], [0]], dtype=np.float32),
                  "edges": np.array([[l]], dtype=np.float32),
                  "receivers": np.array([0]),
                  "senders": np.array([1])
            }


    dynamic_dict = {"globals": np.array([0.], dtype=np.float32),
                  "nodes": np.array([[pole_pos, pole_vel, pole_angle, pole_angvel],
                                     [cart_pos, cart_vel, cart_angle, cart_angvel]], dtype=np.float32),
                  "edges": np.array([[action]], dtype=np.float32),
                  "receivers": np.array([0]),
                  "senders": np.array([1])
            }

    return static_dict, dynamic_dict


gn_monitor = GraphNetsMonitor(name = NAME,
                              env_name = 'CartPole-v1',
                              build_dics_function = build_dics_cartpole,
                              nb_features = 4,
                              lr=1e-3,
                              max_iter = 1000,
                              nb_trajectories = 200)
gn_monitor.work()
