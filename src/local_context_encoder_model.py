import torch
import torch.nn as nn


class LocalContextEncoder(nn.Module):
  def __init__(self, dropout_keep_prob, lstm_size, num_lstm_layers, word_embed_len, context_embed_len):
    super(LocalContextEncoder, self).__init__()
    self.num_lstm_layers = num_lstm_layers
    self.lstm_size = lstm_size
    self.word_embed_len = word_embed_len
    self.context_embed_len = context_embed_len
    self.dropout_keep_prob = dropout_keep_prob
    self.left_lstm = nn.LSTM(input_size=self.word_embed_len,
                             hidden_size=self.lstm_size,
                             dropout=self.dropout_keep_prob,
                             batch_first=True)
    self.right_lstm = nn.LSTM(input_size=self.word_embed_len,
                              hidden_size=self.lstm_size,
                              dropout=self.dropout_keep_prob,
                              batch_first=True)
    self.projection = nn.Linear(2 * self.lstm_size, self.context_embed_len)
    self.relu = nn.ReLU()

  def forward(self, sentence_splits):
    mention_left_sequences = sentence_splits[:, 0]
    mention_right_sequences = sentence_splits[:, 1]
    sentence_embed = torch.cat((self.left_lstm(mention_left_sequences),
                                self.right_lstm(mention_right_sequences)),
                               dim=1)
    return self.relu(self.projection(sentence_embed))
