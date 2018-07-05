import torch
import torch.nn as nn
import torch.sparse as sparse
import pydash as _
from functools import reduce

from data_transformers import pad_batch


class DocumentContextEncoder(nn.Module):
  def __init__(self,
               dropout_keep_prob,
               lstm_size,
               num_lstm_layers,
               word_embed_len,
               context_embed_len,
               pad_vector):
    super(DocumentContextEncoder, self).__init__()
    self.num_lstm_layers = num_lstm_layers
    self.lstm_size = lstm_size
    self.word_embed_len = word_embed_len
    self.context_embed_len = context_embed_len
    self.dropout_keep_prob = dropout_keep_prob
    self.pad_vector = pad_vector
    self.lstm = nn.LSTM(input_size=self.word_embed_len,
                        hidden_size=self.lstm_size,
                        num_layers=self.num_lstm_layers,
                        dropout=self.dropout_keep_prob,
                        bidirectional=True,
                        batch_first=True)
    self.projection = nn.Linear(2 * self.lstm_size, self.context_embed_len)
    self.relu = nn.ReLU()

  def forward(self, embedded_page_contents):
    batch = pad_batch(self.pad_vector, embedded_page_contents)
    output, state_info = self.lstm(batch)
    last_hidden_state = state_info[0][-2:]
    last_hidden_state_stacked = torch.cat([layer_state for layer_state in last_hidden_state], 1)
    return self.relu(self.projection(last_hidden_state_stacked))
