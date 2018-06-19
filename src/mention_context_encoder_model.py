import torch
import torch.nn as nn


class MentionContextEncoder(nn.Module):
  def __init__(self, local_context_embed_len, document_context_embed_len, embed_len):
    super(MentionContextEncoder, self).__init__()
    self.embed_len = embed_len
    self.local_context_embed_len = local_context_embed_len
    self.document_context_embed_len = document_context_embed_len
    self.projection = nn.Linear(self.local_context_embed_len + self.document_context_embed_len,
                                self.embed_len)
    self.relu = nn.ReLU()

  def forward(self, context_embeds):
    local_context_embeds = context_embeds[:, 0]
    document_context_embeds = context_embeds[:, 1]
    context_embeds = torch.cat(local_context_embeds, document_context_embeds, 2)
    return self.relu(self.projection(context_embeds))

  def loss(self, mention_embeds, batch_true_entity_ids, labels_for_batch):
    self.logits = torch.sum(torch.mul(torch.unsqueeze(mention_embeds, 1),
                                      self.entity_embeds(batch_true_entity_ids)),
                            2)
    return self.criterion(self.logits, labels_for_batch)
