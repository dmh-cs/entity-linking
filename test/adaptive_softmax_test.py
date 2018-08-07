import math
from random import shuffle

import torch
from torch import nn

from adaptive_logits import AdaptiveLogits
from adaptive_softmax import AdaptiveSoftmax

def approx_eq(v1, v2, tol=1e-6):
  return all((v1 - v2) < tol * torch.ones(len(v1)))

def test_softmax():
  batch_size = 300
  hidden_size = 200
  vocab_size = 100
  embed_weights = nn.Parameter(torch.Tensor(vocab_size, hidden_size))
  embed_weights.data.normal_(0, 1.0/math.sqrt(hidden_size))
  vocab = nn.Embedding(vocab_size, hidden_size, _weight=embed_weights)
  cutoffs = [20, 30, vocab_size + 1]
  vocab_indexes_by_frequency = list(range(vocab_size))
  shuffle(vocab_indexes_by_frequency)
  vocab_indexes_by_frequency = torch.tensor(vocab_indexes_by_frequency)
  adaptive_logits = AdaptiveLogits(vocab(vocab_indexes_by_frequency), cutoffs)
  adaptive_softmax = AdaptiveSoftmax(adaptive_logits)
  targets= torch.randint(low=0, high=vocab_size, size=[batch_size], dtype=torch.long)
  hidden = torch.randn(batch_size, hidden_size)
  probs = adaptive_softmax(hidden)
  preds = torch.argmax(probs, dim=1)
  probs_vocab = adaptive_softmax(vocab(targets))
  preds_vocab = torch.argmax(probs_vocab, dim=1)
  assert approx_eq(torch.sum(probs, dim=1), torch.ones(probs.shape[0]))
  assert approx_eq(torch.sum(probs_vocab, dim=1), torch.ones(probs.shape[0]))
  assert (preds_vocab - targets).float().norm(p=float("inf")) < (preds - targets).float().norm(p=float("inf"))
