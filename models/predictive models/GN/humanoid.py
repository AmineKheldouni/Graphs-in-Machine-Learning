# -*- coding: utf-8 -*-

from monitor import *
import numpy as np

NAME = 'Humanoid'

if not os.path.exists('./results/'):
    os.mkdir('./results/')
if not os.path.exists('./results/'+NAME):
    os.mkdir('./results/'+NAME)

### BUILD GRAPH
def build_dics_humanoid(m, l, g, state, action):

    [root_z_pos,root_w,root_x,root_y,root_z] = state[0:5]
    root_vels = state[22:29]

    [abdomen_z,abdomen_y,abdomen_x] = state[5:8]
    [abdomen_z_vel,abdomen_y_vel,abdomen_x_vel] = state[29:32]

    [rhip_x,rhip_z,rhip_y,rknee] = state[8:12]
    [rhip_x_vel,rhip_z_vel,rhip_y_vel,rknee_vel] = state[32:36]


    [lhip_x,lhip_z,lhip_y,lknee] = state[12:16]
    [lhip_x_vel,lhip_z_vel,lhip_y_vel,lknee_vel] = state[36:40]

    [rshoulder_1,rshoulder_2,relbow] = state[16:19]
    [rshoulder_1_vel,rshoulder_2_vel,relbow_vel] = state[40:43]

    [lshoulder_1,lshoulder_2,lelbow] = state[19:22]
    [lshoulder_1_vel,lshoulder_2_vel,lelbow_vel] = state[43:46]


    [abdomen_y_act, abdomen_z_act, abdomen_x_act] = action[0:3]
    [rhip_x_act,rhip_z_act,rhip_y_act,rknee_act] = action[3:7]
    [lhip_x_act,lhip_z_act,lhip_y_act,lknee_act] = action[7:11]
    [rshoulder_1_act,rshoulder_2_act,relbow_act] = action[11:14]
    [lshoulder_1_act,lshoulder_2_act,lelbow_act] = action[14:17]


    static_dict = {"globals": np.array([g], dtype=np.float32),
                  "nodes": np.array([[0.] for i in range(9)], dtype=np.float32),
                  "edges": np.array([[l] for i in range(9)], dtype=np.float32),
                  "receivers": np.array([0,1,2,3,4,5,6,7,8], dtype=np.float32),
                  "senders": np.array([0,0,1,0,3,0,5,0,7], dtype=np.float32)
            }


    dynamic_dict = {"globals": np.hstack([[root_z_pos,root_w,root_x,root_y,root_z], root_vels]).astype(np.float32),
                  "nodes": np.array([[abdomen_z,abdomen_y,abdomen_x,abdomen_z_vel,abdomen_y_vel,abdomen_x_vel],
                                     [rhip_x,rhip_z,rhip_y, rhip_x_vel,rhip_z_vel,rhip_y_vel],
                                     [rknee,0.,0.,rknee_vel,0.,0.],
                                     [lhip_x,lhip_z,lhip_y, lhip_x_vel,lhip_z_vel,lhip_y_vel],
                                     [lknee,0.,0.,lknee_vel,0.,0.],
                                     [rshoulder_1,rshoulder_2,0.,rshoulder_1_vel,rshoulder_2_vel,0.],
                                     [relbow,0.,0.,relbow_vel,0.,0.],
                                     [lshoulder_1,lshoulder_2,0.,lshoulder_1_vel,lshoulder_2_vel,0.],
                                     [lelbow,0.,0.,lelbow_vel,0.,0.]],dtype=np.float32),
                  "edges": np.array([[abdomen_y_act, abdomen_z_act, abdomen_x_act],
                                     [rhip_x_act,rhip_z_act,rhip_y_act],
                                     [rknee_act,0.,0.],
                                     [lhip_x_act,lhip_z_act,lhip_y_act],
                                     [lknee_act,0.,0.],
                                     [rshoulder_1_act,rshoulder_2_act,0.],
                                     [relbow_act,0.,0.],
                                     [lshoulder_1_act,lshoulder_2_act,0.],
                                     [lelbow_act,0.,0.]], dtype=np.float32),
                  "receivers": np.array([0,1,2,3,4,5,6,7,8], dtype=np.float32),
                  "senders": np.array([0,0,1,0,3,0,5,0,7], dtype=np.float32)
            }

    return static_dict, dynamic_dict

gn_monitor = GraphNetsMonitor(name = NAME,
                              env_name = 'Humanoid-v2',
                              build_dics_function = build_dics_humanoid,
                              nb_features = 6,
                              lr=5e-4,
                              max_iter = 2000,
                              nb_trajectories = 500,
                              plot_threshold = 200)
gn_monitor.train()
