import torch
import torch.nn as nn

from utils import Identity

from toolz import pipe

def _get_layer(from_size, to_size, dropout_keep_prob, activation=None):
  lin = nn.Linear(from_size, to_size)
  if (activation == None) or type(activation) == nn.ReLU:
    act_name = 'relu'
  elif type(activation) == nn.Tanh:
    act_name = 'tanh'
  elif type(activation) == nn.LeakyReLU:
    act_name = 'leaky_relu'
  else:
    act_name = 'linear'
  nn.init.xavier_uniform_(lin.weight, gain=nn.init.calculate_gain(act_name))
  return [lin,
          nn.ReLU() if activation is None else activation,
          nn.Dropout(1 - dropout_keep_prob)]

class MLP(nn.Module):
  def __init__(self,
               in_dim,
               out_dim,
               hidden_sizes,
               dropout_keep_prob,
               dropout_after_first_layer_only=True):
    super().__init__()
    self.dropout_after_first_layer_only = dropout_after_first_layer_only
    self.layers = nn.ModuleList()
    from_size = in_dim
    for layer_num, to_size in enumerate(hidden_sizes):
      if self.dropout_after_first_layer_only and (layer_num == 0):
        layer_dropout_kp = 1.0
      else:
        layer_dropout_kp = dropout_keep_prob
      self.layers.extend(_get_layer(from_size,
                                    to_size,
                                    layer_dropout_kp))
      from_size = to_size
    self.layers.extend(_get_layer(from_size, out_dim, dropout_keep_prob, activation=Identity()))

  def forward(self, x):
    return pipe(x, *self.layers)
