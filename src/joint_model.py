import torch.nn as nn

from description_encoder_model import DescriptionEncoder
from mention_context_encoder_model import MentionContextEncoder

class JointEncoder(nn.Module):
  def __init__(self, desc_encoder, mention_context_encoder):
    super().__init__()
    self.desc_encoder = desc_encoder
    self.mention_context_encoder = mention_context_encoder

  def forward(self, data):
    embedded_page_contents = data[1]
    desc_embeds = self.desc_encoder(embedded_page_contents)
    mention_context_embeds = self.mention_context_encoder(data)
    return (desc_embeds, mention_context_embeds)

class Stacker(nn.Module):
  def __init__(self):
    super().__init__()
    num_features = 3
    self.linear = nn.Linear(num_features, 1)

  def forward(self, logits, str_sim):
    lin_result = self.linear(torch.cat([logits.reshape(-1),
                                        str_sim.reshape(-1)]))
                                        # prior.reshape(-1)]))
    return lin_result.reshape(*logits.shape)

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
    self.encoder = JointEncoder(self.desc_encoder, self.mention_context_encoder)
    self.calc_scores = Stacker()
