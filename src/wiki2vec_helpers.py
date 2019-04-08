import torch
import torch.nn as nn

import pydash as _

from wikipedia2vec import Wikipedia2Vec

class Wiki2Vec(nn.Module):
  def __init__(self, wiki2vec, device):
    super().__init__()
    self.wiki2vec = wiki2vec
    self.dim_len = self.wiki2vec.syn0.shape[1]
    self.device = device

  def forward(self, idxs):
    return torch.from_numpy(self.wiki2vec.syn0[idxs.reshape(-1)].reshape(*idxs.shape)).to(self.device)

def load_wiki2vec():
  try:
    wiki2vec = Wikipedia2Vec.load('./enwiki_20180420_500d.pkl')
  except:
    wiki2vec = Wikipedia2Vec.load('./enwiki_20180420_100d.pkl')
  return Wiki2Vec(wiki2vec, torch.device('cpu'))
