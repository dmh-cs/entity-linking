import torch.nn as nn

from description_encoder_model import DescriptionEncoder
from mention_context_encoder_model import MentionContextEncoder

class JointModel(nn.Module):
  def __init__(self,
               embed_len,
               context_embed_len,
               word_embed_len,
               local_encoder_lstm_size,
               document_encoder_lstm_size,
               num_lstm_layers,
               dropout_drop_prob,
               entity_embeds,
               word_embedding,
               pad_vector,
               adaptive_logits,
               use_deep_network,
               use_lstm_local,
               num_cnn_local_filters,
               use_cnn_local):
    super().__init__()
    self.entity_embeds = entity_embeds
    self.desc_encoder = DescriptionEncoder(word_embed_len,
                                           entity_embeds,
                                           pad_vector)
    self.mention_context_encoder = MentionContextEncoder(embed_len,
                                                         context_embed_len,
                                                         word_embed_len,
                                                         local_encoder_lstm_size,
                                                         document_encoder_lstm_size,
                                                         num_lstm_layers,
                                                         dropout_drop_prob,
                                                         entity_embeds,
                                                         pad_vector,
                                                         use_deep_network,
                                                         use_lstm_local,
                                                         num_cnn_local_filters,
                                                         use_cnn_local)
    self.desc = adaptive_logits['desc']
    self.mention = adaptive_logits['mention']
    self.word_embedding = word_embedding

  def forward(self, data):
    embedded_page_contents = data[1]
    desc_embeds = self.desc_encoder(embedded_page_contents)
    mention_context_embeds = self.mention_context_encoder(data)
    return (desc_embeds, mention_context_embeds)
