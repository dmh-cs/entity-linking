import torch
import torch.nn as nn

from mlp import MLP
from utils import is_odd

class MLPSkips(nn.Module):
  def __init__(self,
               in_dim,
               out_dim,
               hidden_size,
               num_hidden,
               dropout_keep_prob,
               dropout_after_first_layer_only=True):
    super().__init__()
    assert is_odd(num_hidden), 'Resnet blocks need an odd number of hidden layers'
    hidden_sizes = [hidden_size for i in range(num_hidden)]
    self.num_hidden = num_hidden
    self.hidden_size = hidden_size
    self.mlp = MLP(in_dim,
                   out_dim,
                   hidden_sizes,
                   dropout_keep_prob,
                   dropout_after_first_layer_only=dropout_after_first_layer_only)

  def forward(self, x):
    outs = [x]
    for layer_num, layer in enumerate(self.mlp.layers):
      if (not is_odd(layer_num)) and (layer_num not in [0, self.num_hidden - 1]):
        outs.append(layer(outs[-1]) + outs[-2])
      else:
        outs.append(layer(outs[-1]))
    return outs[-1]
