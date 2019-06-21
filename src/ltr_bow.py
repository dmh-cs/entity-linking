import torch
import torch.nn as nn

from mlp import MLP


def get_model(model_params, train_params):
  return LtRBoW(hidden_sizes=model_params.hidden_sizes,
                dropout_keep_prob=train_params.dropout_keep_prob,
                tanh_final_layer=train_params.use_hinge)

class LtRBoW(nn.Module):
  def __init__(self, hidden_sizes, dropout_keep_prob=0.5, tanh_final_layer=False):
    super().__init__()
    self.hidden_sizes = hidden_sizes
    self.dropout_keep_prob = dropout_keep_prob
    self.num_features = len(['mention_tfidf',
                             'cand_token_cnt',
                             'mention_token_cnt',
                             'page_tfidf',
                             'page_token_cnt',
                             'str_sim',
                             'prior',
                             'times_mentioned',
                             'mention_wiki2vec_dot',
                             'page_wiki2vec_dot',
                             'mention_wiki2vec_dot_norm',
                             'page_wiki2vec_dot_norm'])
    self.tanh_final_layer = tanh_final_layer
    self.mlp = MLP(self.num_features, 1, self.hidden_sizes, dropout_keep_prob)

  def forward(self, features):
    if self.tanh_final_layer:
      return torch.tanh(self.mlp(features).reshape(-1))
    else:
      return self.mlp(features).reshape(-1)
