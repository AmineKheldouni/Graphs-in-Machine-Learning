import sonnet as snt
import keras.utils as np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, UpSampling2D, Activation
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical

from graph_nets import blocks
from graph_nets import utils_tf
from graph_nets.demos import models
from graph_nets import modules


### CREATE MODEL CLASS

class MLPGraphNetwork(snt.AbstractModule):
  def __init__(self, n_latent = 128, n_hidden_layers = 2, n_output = 2, name="MLPGraphNetwork"):
    super(MLPGraphNetwork, self).__init__(name=name)
    self.n_latent = n_latent
    self.n_hidden_layers = n_hidden_layers
    self.n_output = n_output
    with self._enter_variable_scope():
      self._network = modules.GraphNetwork(self.make_mlp_model, self.make_mlp_model,
                                           self.make_mlp_model)


  def make_mlp_model(self):
      return snt.Sequential([
          snt.nets.MLP([self.n_latent] * self.n_hidden_layers, activate_final=True),
          snt.nets.MLP([self.n_output], activate_final = False)
      ])

  def _build(self, inputs):
    return self._network(inputs)


class ForwardModel(snt.AbstractModule):

    def __init__(self, name, n_latent_GN = 128, n_hidden_layers_GN = 2, n_output_latent = 2, n_output_final = 2):
        super(ForwardModel, self).__init__(name=name)
        with self._enter_variable_scope():
            self._network_1 = MLPGraphNetwork(n_latent_GN, n_hidden_layers_GN, n_output_latent)
            self._network_2 = MLPGraphNetwork(n_latent_GN, n_hidden_layers_GN, n_output_final)

    def _build(self, graph_input):
        graph_latent = self._network_1(graph_input)
        graph_concat = utils_tf.concat([graph_input, graph_latent], axis=1)
        output = self._network_2(graph_concat)
        return output
