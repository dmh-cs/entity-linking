import torch
import torch.nn as nn

from mlp import MLP


class LtRBoW(nn.Module):
  def __init__(self, hidden_sizes, dropout_keep_prob=0.5):
    super().__init__()
    self.hidden_sizes = hidden_sizes
    self.dropout_keep_prob = dropout_keep_prob
    self.num_features = len(['mention_tfidf', 'page_tfidf', 'str_sim', 'prior', 'times_mentioned'])
    self.mlp = MLP(self.num_features, 1, self.hidden_sizes, dropout_keep_prob)

  def forward(self, features):
    return self.mlp(features).reshape(-1)
