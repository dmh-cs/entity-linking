from random import shuffle
import math

import torch
from torch import nn

from adaptive_logits import AdaptiveLogits

def test_loss():
  batch_size = 5
  hidden_size = 8
  vocab_size = 100
  vocab = nn.Embedding(vocab_size, hidden_size)
  cutoffs = [20, 30, vocab_size + 1]
  vocab_indexes_by_frequency = list(range(vocab_size))
  shuffle(vocab_indexes_by_frequency)
  vocab_indexes_by_frequency = torch.tensor(vocab_indexes_by_frequency)
  adaptive_logits = AdaptiveLogits(vocab(vocab_indexes_by_frequency), cutoffs)
  hidden = torch.randn(batch_size, hidden_size)
  targets = torch.randint(low=0, high=vocab_size + 1, size=[batch_size], dtype=torch.long)
  all_logits = adaptive_logits(hidden, targets)
  loss = adaptive_logits.loss(all_logits, targets)
  assert loss

def test_loss_check_loss():
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
  hidden = torch.randn(batch_size, hidden_size)
  targets = torch.randint(low=0, high=vocab_size, size=[batch_size], dtype=torch.long)
  all_logits = adaptive_logits(hidden, targets)
  loss = adaptive_logits.loss(all_logits, targets)
  all_logits_vocab = adaptive_logits(vocab(targets), targets)
  loss_vocab = adaptive_logits.loss(all_logits_vocab, targets)
  assert loss_vocab < loss
