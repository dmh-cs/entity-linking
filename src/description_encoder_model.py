import torch
import torch.nn as nn
import torch.nn.functional as F


class DescriptionEncoder(nn.Module):
  def __init__(self, embed_len, entity_embeds):
    super(DescriptionEncoder, self).__init__()
    self.entity_embeds = entity_embeds
    self.embed_len = embed_len
    self.kernel_size = 5
    self.dropout_keep_prob = 0.6
    self.conv = nn.Conv2d(1, self.embed_len, self.kernel_size, stride=1, padding=0)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(p=self.dropout_keep_prob)
    self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
    self.criterion = nn.CrossEntropyLoss()

  def forward(self, descriptions):
    return torch.squeeze(self.global_avg_pooling(self.dropout(self.relu(self.conv(descriptions)))))

  def loss(self, desc_embeds, batch_true_entity_ids):
    labels_for_batch = torch.tensor(range(desc_embeds.shape[0]))
    self.logits = torch.sum(torch.mul(torch.unsqueeze(desc_embeds, 1),
                                      self.entity_embeds(batch_true_entity_ids)),
                            2)
    return self.criterion(self.logits, labels_for_batch)
