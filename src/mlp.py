import torch
import torch.nn as nn

from utils import Identity

from toolz import pipe

def _get_layer(from_size, to_size, dropout_keep_prob, activation=None):
  return [nn.Linear(from_size, to_size),
          nn.ReLU() if activation is None else activation,
          nn.Dropout(1 - dropout_keep_prob)]

class MLP(nn.Module):
  def __init__(self, in_dim, out_dim, hidden_layer_sizes, dropout_keep_prob):
    super().__init__()
    self.layers = nn.ModuleList()
    from_size = in_dim
    for to_size in hidden_layer_sizes:
      self.layers.extend(_get_layer(from_size,
                                    to_size,
                                    dropout_keep_prob))
      from_size = to_size
    self.layers.extend(_get_layer(from_size, out_dim, dropout_keep_prob, activation=Identity()))

  def forward(self, x):
    return pipe(x, *self.layers)
