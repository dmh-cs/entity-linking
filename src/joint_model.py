import torch.nn as nn

from description_encoder_model import DescriptionEncoder
from mention_context_encoder_model import MentionContextEncoder

class JointModel(nn.Module):
  def __init__(self,
               embed_len,
               context_embed_len,
               word_embed_len,
               lstm_size,
               num_lstm_layers,
               dropout_keep_prob,
               entity_embeds,
               pad_vector):
    super().__init__()
    self.desc_encoder = DescriptionEncoder(word_embed_len,
                                           entity_embeds,
                                           pad_vector)
    self.mention_context_encoder = MentionContextEncoder(embed_len,
                                                         context_embed_len,
                                                         word_embed_len,
                                                         lstm_size,
                                                         num_lstm_layers,
                                                         dropout_keep_prob,
                                                         entity_embeds,
                                                         pad_vector)

  def forward(self, data):
    embedded_page_contents = data[1]
    desc_embeds = self.desc_encoder(embedded_page_contents)
    mention_context_embeds = self.mention_context_encoder(data)
    return (desc_embeds, mention_context_embeds)

  def loss(self, encoded, candidate_entity_ids, labels_for_batch):
    desc_embeds, mention_context_embeds = encoded
    desc_loss = self.desc_encoder.loss(desc_embeds,
                                       candidate_entity_ids,
                                       labels_for_batch)
    mention_loss = self.mention_context_encoder.loss(mention_context_embeds,
                                                     candidate_entity_ids,
                                                     labels_for_batch)
    return desc_loss + mention_loss
