import torch
import torch.nn as nn
import torch.nn.functional as F


class DescriptionEncoder(nn.Module):
  def __init__(self):
    super(DescriptionEncoder, self).__init__()
    self.desc_embed_len = 150
    self.kernel_size = 5
    self.dropout_keep_prob = 0.6
    self.conv = nn.Conv2d(1, self.desc_embed_len, self.kernel_size, stride=[1, 1, 1, 1], padding=0)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(p=self.dropout_keep_prob)
    self.global_avg_pooling = nn.AvgPool1d(self.kernel_size - self.desc_embed_len)
    self.criterion = nn.CrossEntropyLoss()

  def forward(self, descriptions):
    return self.global_avg_pooling(self.dropout(self.relu(self.conv(descriptions))))

  def loss(self, desc_embeds, true_entity_ids):
    self.logits = torch.sum(torch.mul(desc_embeds, self.entity_embeds), 1)
    return self.criterion(self.logits, true_entity_ids)
