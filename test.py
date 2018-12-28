import time

import graph_nets
from matplotlib import pyplot as plt
import numpy as np
import pandas
import sonnet as snt
import tensorflow as tf
import gym
from dm_control import suite
from dm_control import viewer
print("Imported libraries !")

# Run: export MUJOCO_GL=glfw in Terminal before running this file.


SEED = 1
np.random.seed(SEED)
tf.set_random_seed(SEED)


print("Creating environment ...")
# Load one task:
env = suite.load(domain_name="acrobot", task_name="swingup")

print("Environment created !")

# Action set
action_spec = env.action_spec()
# Define a uniform random policy.
def random_policy(time_step):
  del time_step  # Unused.
  return np.random.uniform(low=action_spec.minimum,
                           high=action_spec.maximum,
                           size=action_spec.shape)

print("Rendering the environment ...")
# Launch the viewer application.
viewer.launch(env, policy=random_policy)
env.close()
