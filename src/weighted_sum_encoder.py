import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedSumEncoder(nn.Module):
  def __init__(self, word_embeds, use_cnts=False, idf=None):
    super().__init__()
    self.word_embeds = word_embeds
    self.use_cnts = use_cnts
    self.idf = idf
    self.weights = nn.Embedding(len(word_embeds.weight), 1)
    nn.init.xavier_normal_(self.weights.weight.data)
    if self.idf is not None:
      for idx, weight in self.idf.items():
        self.weights.weight.data[idx] = weight

  def forward(self, desc):
    if self.use_cnts:
      terms, cnts = desc
      terms = terms.long()
      token_weights = self.weights(terms).reshape(cnts.shape) + torch.log(cnts.float())
      normalized_weights = F.softmax(token_weights, 1)
      document_tokens = self.word_embeds(terms)
      doc_vecs = torch.sum(normalized_weights.unsqueeze(2) * document_tokens, 1)
      return doc_vecs
    else:
      desc_tokens = self.word_embeds(desc)
      token_weights = self.weights(desc)
      normalized_weights = F.softmax(token_weights, 1)
      return torch.sum(normalized_weights * desc_tokens, 1)
