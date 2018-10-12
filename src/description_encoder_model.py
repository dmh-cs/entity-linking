from toolz import pipe
import pydash as _
import torch
import torch.nn as nn

from data_transformers import pad_batch


class DescriptionEncoder(nn.Module):
  def __init__(self, word_embed_len, entity_embeds, pad_vector):
    super(DescriptionEncoder, self).__init__()
    self.pad_vector = pad_vector
    self.kernel_size = 5
    self.dropout_drop_prob = 0.4
    desc_embed_len = entity_embeds.weight.shape[1]
    self.conv = nn.Conv1d(word_embed_len, desc_embed_len, self.kernel_size, stride=1, padding=0)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(p=self.dropout_drop_prob)
    self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)

  def forward(self, embedded_page_contents):
    desc_embeds = pad_batch(self.pad_vector,
                            [embeds[:100] for embeds in embedded_page_contents],
                            min_len=100)
    encoded = pipe(desc_embeds,
                   # lambda embed: torch.transpose(embed, 1, 2),
                   self.conv,
                   self.relu,
                   self.dropout,
                   self.global_avg_pooling,
                   torch.squeeze)
    return encoded / torch.norm(encoded, 2, 1).unsqueeze(1)
