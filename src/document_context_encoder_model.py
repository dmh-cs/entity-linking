import torch
import torch.nn as nn
import torch.sparse as sparse


class DocumentContextEncoder(nn.Module):
  def __init__(self, num_mentions, document_context_embed_len):
    super(DocumentContextEncoder, self).__init__()
    self.num_mentions = num_mentions
    self.document_context_embed_len = document_context_embed_len
    self.projection = nn.Linear(self.num_mentions, self.document_context_embed_len)
    self.relu = nn.ReLU()

  def forward(self, document_mention_indices):
    batch_size = document_mention_indices.shape[0]
    mention_surfaces = sparse.LongTensor(document_mention_indices,
                                         torch.ones(document_mention_indices.shape),
                                         (batch_size, self.num_mentions))
    return self.relu(self.projection(mention_surfaces))
