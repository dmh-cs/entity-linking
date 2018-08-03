import torch
import torch.nn as nn
from document_context_encoder_model import DocumentContextEncoder
from local_context_encoder_model import LocalContextEncoder


class MentionContextEncoder(nn.Module):
  def __init__(self,
               embed_len,
               context_embed_len,
               word_embed_len,
               local_encoder_lstm_size,
               document_encoder_lstm_size,
               num_lstm_layers,
               dropout_keep_prob,
               entity_embeds,
               pad_vector):
    super(MentionContextEncoder, self).__init__()
    self.entity_embeds = entity_embeds
    self.local_context_encoder = LocalContextEncoder(dropout_keep_prob,
                                                     local_encoder_lstm_size,
                                                     num_lstm_layers,
                                                     word_embed_len,
                                                     context_embed_len)
    self.document_context_encoder = DocumentContextEncoder(document_encoder_lstm_size,
                                                           word_embed_len,
                                                           context_embed_len,
                                                           pad_vector)
    self.projection = nn.Linear(2 * context_embed_len, embed_len)
    self.relu = nn.ReLU()

  def forward(self, data):
    sentence_splits = data[0]
    entity_page_mentions = data[2]
    local_context_embeds = self.local_context_encoder(sentence_splits)
    document_context_embeds = self.document_context_encoder(entity_page_mentions)
    context_embeds = torch.cat((local_context_embeds, document_context_embeds), 1)
    return self.relu(self.projection(context_embeds))
