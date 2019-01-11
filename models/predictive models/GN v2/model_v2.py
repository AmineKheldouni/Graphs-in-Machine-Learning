import sonnet as snt
import keras.utils as np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, UpSampling2D, Activation
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical

from graph_nets import blocks
from graph_nets import utils_tf, utils_np
from graph_nets.demos import models
from graph_nets import modules


### CREATE MODEL CLASS

class MLPGraphNetwork(snt.AbstractModule):
  def __init__(self, n_latent = 128, n_hidden_layers = 2, n_output_nodes = 2, n_output_edges = 2, n_output_globals = 2, name="MLPGraphNetwork"):
    super(MLPGraphNetwork, self).__init__(name=name)
    self.n_latent = n_latent
    self.n_hidden_layers = n_hidden_layers
    self.n_output_nodes = n_output_nodes
    self.n_output_edges = n_output_edges
    self.n_output_globals = n_output_globals
    with self._enter_variable_scope():
      self._network = modules.GraphNetwork(self.make_mlp_model_edges, self.make_mlp_model_nodes,
                                           self.make_mlp_model_globals)


  def make_mlp_model_nodes(self):
      return snt.Sequential([
          snt.nets.MLP([self.n_latent] * self.n_hidden_layers, activate_final=True),
          snt.nets.MLP([self.n_output_nodes], activate_final = False)
      ])
    
  def make_mlp_model_edges(self):
      return snt.Sequential([
          snt.nets.MLP([self.n_latent] * self.n_hidden_layers, activate_final=True),
          snt.nets.MLP([self.n_output_edges], activate_final = False)
      ])
    
  def make_mlp_model_globals(self):
      return snt.Sequential([
          snt.nets.MLP([self.n_latent] * self.n_hidden_layers, activate_final=True),
          snt.nets.MLP([self.n_output_globals], activate_final = False)
      ])

  def _build(self, inputs):
    return self._network(inputs)


class ForwardModel(snt.AbstractModule):

    def __init__(self, name, n_latent_GN = 128, n_hidden_layers_GN = 2, n_output_latent = 2, n_output_nodes = 2, n_output_edges = 2, n_output_globals = 2):
        super(ForwardModel, self).__init__(name=name)
        with self._enter_variable_scope():
            self._network_1 = MLPGraphNetwork(n_latent_GN, n_hidden_layers_GN, n_output_latent, n_output_edges, n_output_globals)
            self._network_2 = MLPGraphNetwork(n_latent_GN, n_hidden_layers_GN, n_output_nodes, n_output_edges, n_output_globals)

    def _build(self, graph_input):
        graph_latent = self._network_1(graph_input)
        graph_concat = utils_tf.concat([graph_input, graph_latent], axis=1)
        output = self._network_2(graph_concat)
        return output

class RecurrentModelInference(snt.AbstractModule):
    
    def __init__(self, name, n_latent_GN = 128, n_hidden_layers_GN = 2, n_output_latent = 2, n_output_nodes = 2, n_output_edges = 2, n_output_globals = 2, T = 5):
        
        super(RecurrentModelInference, self).__init__(name=name)
        
        self.T = T
        self.n_output_nodes = n_output_nodes
        self.n_output_edges = n_output_edges
        self.n_output_globals = n_output_globals
        
        with self._enter_variable_scope():
            self._first_hidden_net = MLPGraphNetwork(n_latent_GN, n_hidden_layers_GN, n_output_nodes, n_output_edges, n_output_globals)
            self._recurrent_core = MLPGraphNetwork(n_latent_GN, n_hidden_layers_GN, 2*n_output_nodes, 2*n_output_edges, 2*n_output_globals)
            self._forward_model = ForwardModel(name+"_forward", n_latent_GN, n_hidden_layers_GN, n_output_latent, n_output_nodes, n_output_edges, n_output_globals)
        
    def _build(self, all_graphs_input):
        
        n_examples = int(all_graphs_input.globals.shape[0]) - self.T + 1
        
        #for batch in range(int(all_graphs_input.globals.shape[0])-self.T):
            
            #T_graphs_input = utils_np.get_graph(all_graphs_input, slice(batch, batch+self.T))

        for t in range(self.T):
            
            #print(utils_np.graphs_tuple_to_data_dicts(all_graphs_input))
            graph = utils_tf.get_graph(all_graphs_input, slice(t, n_examples+t))
            if t == 0:
                graph_hidden = self._first_hidden_net(graph)
                
            #print(graph)
                
            graph_concat = utils_tf.concat([graph, graph_hidden], axis=1)
            graph_concat_new = self._recurrent_core(graph_concat)
            
            #print(graph_concat_new)
            
            graph = graph.replace(nodes=graph_concat_new.nodes[:, :self.n_output_nodes],
                                  globals=graph_concat_new.globals[:, :self.n_output_globals],
                                  edges=graph_concat_new.edges[:, :self.n_output_edges])
            
            graph_hidden = graph_hidden.replace(nodes=graph_concat_new.nodes[:, self.n_output_nodes:],
                                                globals=graph_concat_new.globals[:, self.n_output_globals:],
                                                edges=graph_concat_new.edges[:, self.n_output_edges:])
            
            if t == self.T - 1:

                graph_concat_output = utils_tf.concat([graph, graph_hidden], axis=1)
                output_graph = self._forward_model(graph_concat_output)
                break
        
        
        return output_graph
                    
                
                
                
                
                
            
    
            
        