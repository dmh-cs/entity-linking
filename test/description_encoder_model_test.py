import torch

from description_encoder_model import DescriptionEncoder

def test_description_encoder_forward():
  embed_len = 10
  word_embed_len = 15
  num_entities = 20
  batch_size = 2
  desc_len = 9
  desc_enc = DescriptionEncoder(embed_len, torch.nn.Embedding(num_entities,
                                                              embed_len,
                                                              _weight=torch.rand((num_entities, embed_len))))
  descriptions = torch.rand((batch_size, 1, word_embed_len, desc_len))
  desc_embeds = desc_enc(descriptions)
  assert desc_embeds.shape == torch.Size([2, embed_len])

def test_description_encoder_loss():
  embed_len = 10
  word_embed_len = 15
  num_entities = 20
  batch_size = 2
  desc_len = 9
  desc_enc = DescriptionEncoder(embed_len, torch.nn.Embedding(num_entities,
                                                              embed_len,
                                                              _weight=torch.rand((num_entities, embed_len))))
  descriptions = torch.rand((batch_size, 1, word_embed_len, desc_len))
  desc_embeds = desc_enc(descriptions)
  loss = desc_enc.loss(desc_embeds,
                       torch.randint(0, num_entities, (batch_size,), dtype=torch.long))
  assert isinstance(loss, torch.Tensor)
  assert loss.shape == torch.Size([])
