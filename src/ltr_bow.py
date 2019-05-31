import torch
import torch.nn as nn

from mlp import MLP


class LtRBoW(nn.Module):
  def __init__(self, hidden_size=100, dropout_keep_prob=0.5):
    self.hidden_size = hidden_size
    self.dropout_keep_prob = dropout_keep_prob
    self.num_features = len(['str_sim', 'prior', 'tfidf'])
    self.mlp = MLP(self.num_features, 1, [hidden_size], dropout_keep_prob)

  def forward(self, features):
    return self.mlp(features)
