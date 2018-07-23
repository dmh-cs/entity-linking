from toolz import compose
import pydash as _
import torch
import torch.nn as nn

from data_transformers import pad_batch


class DescriptionEncoder(nn.Module):
  def __init__(self, word_embed_len, entity_embeds, pad_vector):
    super(DescriptionEncoder, self).__init__()
    self.entity_embeds = entity_embeds
    self.pad_vector = pad_vector
    self.kernel_size = 5
    self.dropout_keep_prob = 0.6
    desc_embed_len = entity_embeds.weight.shape[1]
    self.conv = nn.Conv1d(word_embed_len, desc_embed_len, self.kernel_size, stride=1, padding=0)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(p=self.dropout_keep_prob)
    self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
    self.criterion = nn.CrossEntropyLoss()

  def forward(self, embedded_page_contents):
    desc_embeds = pad_batch(self.pad_vector,
                            [embeds[:100] for embeds in embedded_page_contents],
                            min_len=100)
    fn = compose(torch.squeeze,
                 self.global_avg_pooling,
                 self.dropout,
                 self.relu,
                 self.conv,
                 _.partial_right(torch.transpose, 1, 2))
    return fn(desc_embeds)

  def loss(self, desc_embeds, candidate_entity_ids, labels_for_batch):
    self.logits = torch.sum(torch.mul(torch.unsqueeze(desc_embeds, 1),
                                      self.entity_embeds(candidate_entity_ids)),
                            2)
    return self.criterion(self.logits, labels_for_batch)
