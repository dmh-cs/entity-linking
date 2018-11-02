from toolz import pipe

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
from data_transformers import get_splits_and_order


class LocalContextEncoder(nn.Module):
  def __init__(self,
               dropout_drop_prob,
               lstm_size,
               num_lstm_layers,
               word_embed_len,
               context_embed_len,
               use_deep_network=True,
               num_cnn_local_filters=None,
               use_cnn_local=False):
    super(LocalContextEncoder, self).__init__()
    self.num_lstm_layers = num_lstm_layers
    self.lstm_size = lstm_size
    self.word_embed_len = word_embed_len
    self.context_embed_len = context_embed_len
    self.dropout_drop_prob = dropout_drop_prob
    self.use_cnn_local = use_cnn_local
    self.num_cnn_local_filters = num_cnn_local_filters
    if self.use_cnn_local:
      self.left_cnn = nn.Conv1d(self.word_embed_len, num_cnn_local_filters, 5)
      self.left_relu = nn.ReLU()
      self.left_pool = nn.AdaptiveMaxPool1d(1)
      self.right_cnn = nn.Conv1d(self.word_embed_len, num_cnn_local_filters, 5)
      self.right_relu = nn.ReLU()
      self.right_pool = nn.AdaptiveMaxPool1d(1)
    else:
      self.left_lstm = nn.LSTM(input_size=self.word_embed_len,
                               hidden_size=self.lstm_size,
                               num_layers=self.num_lstm_layers,
                               dropout=self.dropout_drop_prob,
                               bidirectional=True)
      self.right_lstm = nn.LSTM(input_size=self.word_embed_len,
                                hidden_size=self.lstm_size,
                                num_layers=self.num_lstm_layers,
                                dropout=self.dropout_drop_prob,
                                bidirectional=True)
    self.use_deep_network = use_deep_network
    if self.use_deep_network:
      self.projection = nn.Linear(2 * 2 * self.lstm_size * self.num_lstm_layers,
                                  self.context_embed_len)
    elif self.use_cnn_local:
      self.projection = nn.Linear(2 * self.num_cnn_local_fiters, self.context_embed_len)
    else:
      self.projection = nn.Linear(self.word_embed_len * 2,
                                  self.context_embed_len)
    self.relu = nn.ReLU()

  def forward(self, sentence_splits):
    left_splits, left_order = get_splits_and_order(sentence_splits[0])
    right_splits, right_order = get_splits_and_order(sentence_splits[1])
    if self.use_deep_network:
      left_output, left_state_info = self.left_lstm(left_splits)
      right_output, right_state_info = self.right_lstm(right_splits)
      left_last_hidden_state = torch.cat([left_state_info[0][i] for i in range(left_state_info[0].shape[0])],
                                         1)
      right_last_hidden_state = torch.cat([right_state_info[0][i] for i in range(right_state_info[0].shape[0])],
                                          1)
      left = left_last_hidden_state[left_order]
      right = right_last_hidden_state[right_order]
    elif self.use_cnn_local:
      left_tokens = pad_packed_sequence(left_splits,
                                        padding_value=0,
                                        batch_first=True)[0][left_order]
      right_tokens = pad_packed_sequence(right_splits,
                                        padding_value=0,
                                        batch_first=True)[0][right_order]
      left = pipe(left_tokens,
                  lambda batch: torch.transpose(batch, 1, 2),
                  self.left_cnn,
                  self.left_relu,
                  self.left_pool,
                  torch.squeeze)
      right = pipe(right_tokens,
                   lambda batch: torch.transpose(batch, 1, 2),
                   self.right_cnn,
                   self.right_relu,
                   self.right_pool,
                   torch.squeeze)
    else:
      left_tokens = pad_packed_sequence(left_splits,
                                        padding_value=0,
                                        batch_first=True)[0][left_order]
      right_tokens = pad_packed_sequence(right_splits,
                                        padding_value=0,
                                        batch_first=True)[0][right_order]
      left = left_tokens.sum(1) / left_tokens.shape[1]
      right = right_tokens.sum(1) / right_tokens.shape[1]
    sentence_embed = torch.cat((left, right),
                               dim=1)
    encoded = self.relu(self.projection(sentence_embed))
    return encoded
