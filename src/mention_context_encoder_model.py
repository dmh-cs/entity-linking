import torch
import torch.nn as nn
from document_context_encoder_model import DocumentContextEncoder
from local_context_encoder_model import LocalContextEncoder


class MentionContextEncoder(nn.Module):
  def __init__(self,
               embed_len,
               context_embed_len,
               word_embed_len,
               num_mentions,
               lstm_size,
               num_lstm_layers,
               dropout_keep_prob,
               entity_embeds):
    super(MentionContextEncoder, self).__init__()
    self.entity_embeds = entity_embeds
    self.local_context_encoder = LocalContextEncoder(dropout_keep_prob,
                                                     lstm_size,
                                                     num_lstm_layers,
                                                     word_embed_len,
                                                     context_embed_len)
    # self.document_context_encoder = DocumentContextEncoder(num_mentions, context_embed_len)
    # self.projection = nn.Linear(2 * context_embed_len, embed_len)
    self.projection = nn.Linear(context_embed_len, embed_len)
    self.relu = nn.ReLU()
    self.criterion = nn.CrossEntropyLoss()

  def forward(self, data):
    sentence_splits = data[0]
    document_mention_indices = data[1]
    local_context_embeds = self.local_context_encoder(sentence_splits)
    # document_context_embeds = self.document_context_encoder(document_mention_indices)
    # context_embeds = torch.cat((local_context_embeds, document_context_embeds), 1)
    context_embeds = local_context_embeds
    return self.relu(self.projection(context_embeds))

  def loss(self, mention_embeds, candidate_entity_ids, labels_for_batch):
    self.logits = torch.sum(torch.mul(torch.unsqueeze(mention_embeds, 1),
                                      self.entity_embeds(candidate_entity_ids)),
                            2)
    return self.criterion(self.logits, labels_for_batch)
