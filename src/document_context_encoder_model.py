import pydash as _
import torch
import torch.nn as nn

from data_transformers import pad_batch


class DocumentContextEncoder(nn.Module):
  def __init__(self,
               lstm_size,
               word_embed_len,
               context_embed_len,
               pad_vector):
    super(DocumentContextEncoder, self).__init__()
    self.lstm_size = lstm_size
    self.word_embed_len = word_embed_len
    self.context_embed_len = context_embed_len
    self.pad_vector = pad_vector
    self.lstm = nn.LSTM(input_size=self.word_embed_len,
                        hidden_size=self.lstm_size,
                        batch_first=True)
    self.projection = nn.Linear(self.lstm_size, self.context_embed_len)
    self.relu = nn.ReLU()

  def forward(self, entity_page_mentions):
    batch = pad_batch(self.pad_vector, entity_page_mentions)
    output, state_info = self.lstm(batch)
    last_hidden_state = state_info[0][-2:]
    last_hidden_state_stacked = torch.cat([layer_state for layer_state in last_hidden_state], 1)
    return self.relu(self.projection(last_hidden_state_stacked))
