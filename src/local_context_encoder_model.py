import torch
import torch.nn as nn
from data_transformers import get_splits_and_order


class LocalContextEncoder(nn.Module):
  def __init__(self,
               dropout_keep_prob,
               lstm_size,
               num_lstm_layers,
               word_embed_len,
               context_embed_len):
    super(LocalContextEncoder, self).__init__()
    self.num_lstm_layers = num_lstm_layers
    self.lstm_size = lstm_size
    self.word_embed_len = word_embed_len
    self.context_embed_len = context_embed_len
    self.dropout_keep_prob = dropout_keep_prob
    self.left_lstm = nn.LSTM(input_size=self.word_embed_len,
                             hidden_size=self.lstm_size,
                             num_layers=self.num_lstm_layers,
                             dropout=self.dropout_keep_prob,
                             batch_first=True)
    self.right_lstm = nn.LSTM(input_size=self.word_embed_len,
                              hidden_size=self.lstm_size,
                              num_layers=self.num_lstm_layers,
                              dropout=self.dropout_keep_prob,
                              batch_first=True)
    self.projection = nn.Linear(2 * self.lstm_size, self.context_embed_len)
    self.relu = nn.ReLU()

  def forward(self, sentence_splits):
    left_splits, left_order = get_splits_and_order(sentence_splits[0])
    right_splits, right_order = get_splits_and_order(sentence_splits[1])
    left_output, left_state_info = self.left_lstm(left_splits)
    right_output, right_state_info = self.right_lstm(right_splits)
    left_last_hidden_state = left_state_info[0][-1]
    right_last_hidden_state = right_state_info[0][-1]
    unsorted_left = left_last_hidden_state[left_order]
    unsorted_right = right_last_hidden_state[right_order]
    sentence_embed = torch.cat((unsorted_left, unsorted_right),
                               dim=1)
    return self.relu(self.projection(sentence_embed))
