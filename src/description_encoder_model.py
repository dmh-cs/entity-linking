import torch
import torch.nn as nn


class DescriptionEncoder(nn.Module):
  def __init__(self, word_embed_len, entity_embeds):
    super(DescriptionEncoder, self).__init__()
    self.entity_embeds = entity_embeds
    self.kernel_size = 5
    self.dropout_keep_prob = 0.6
    desc_embed_len = entity_embeds.weight.shape[1]
    self.conv = nn.Conv1d(word_embed_len, desc_embed_len, self.kernel_size, stride=1, padding=0)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(p=self.dropout_keep_prob)
    self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
    self.criterion = nn.CrossEntropyLoss()

  def forward(self, descriptions):
    return torch.squeeze(self.global_avg_pooling(self.dropout(self.relu(self.conv(descriptions)))))

  def loss(self, desc_embeds, batch_true_entity_ids, labels_for_batch):
    self.logits = torch.sum(torch.mul(torch.unsqueeze(desc_embeds, 1),
                                      self.entity_embeds(batch_true_entity_ids)),
                            2)
    return self.criterion(self.logits, labels_for_batch)
