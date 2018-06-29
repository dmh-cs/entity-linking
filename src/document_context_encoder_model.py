import torch
import torch.nn as nn
import torch.sparse as sparse
import pydash as _


class DocumentContextEncoder(nn.Module):
  def __init__(self, num_mentions, context_embed_len):
    super(DocumentContextEncoder, self).__init__()
    self.num_mentions = num_mentions
    self.context_embed_len = context_embed_len
    self.projection = nn.Linear(self.num_mentions, self.context_embed_len)
    self.relu = nn.ReLU()

  def forward(self, document_mention_indices):
    batch_size = len(document_mention_indices)
    first_indices = []
    for elem_num, elem_indices in enumerate(document_mention_indices):
      first_indices.extend([elem_num] * len(elem_indices))
    indices = (first_indices, _.flatten(document_mention_indices))
    mention_surfaces = sparse.FloatTensor(torch.tensor(indices, dtype=torch.long),
                                         torch.ones(len(first_indices)),
                                         (batch_size, self.num_mentions))
    return self.relu(self.projection(mention_surfaces))
