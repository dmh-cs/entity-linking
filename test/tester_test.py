from unittest.mock import Mock
import pytest
import string

import pydash as _
import torch
from torch.utils.data.sampler import BatchSampler, RandomSampler

import torch.nn as nn
import utils as u

import tester as t

mock_call = Mock.__call__

@pytest.fixture(scope="module")
def mock_restore():
  yield None
  Mock.__call__ = mock_call

def get_mock_model(vector_to_return):
  model = Mock()
  model.to = lambda device: model
  Mock.__call__ = lambda self, x: torch.unsqueeze(vector_to_return, 0)
  return model

def test_tester(monkeypatch, mock_restore):
  dataset = [{'label': 0, 'sentence_splits': [['a', 'b', 'c'], ['c', 'd']], 'candidates': torch.tensor([0, 1])},
             {'label': 2, 'sentence_splits': [['a', 'b', 'c'], ['c', 'd']], 'candidates': torch.tensor([2, 1])},
             {'label': 1, 'sentence_splits': [['a', 'b', 'c'], ['c', 'd']], 'candidates': torch.tensor([3, 1])}]
  num_entities = 10
  embed_len = 200
  batch_size = 3
  entity_embeds = nn.Embedding(num_entities,
                               embed_len,
                               _weight=torch.randn((num_entities, embed_len)))
  embedding_lookup = dict(zip(string.ascii_lowercase,
                              [torch.tensor(i) for i, char in enumerate(string.ascii_lowercase)]))
  vector_to_return = entity_embeds(torch.tensor(1))
  model = get_mock_model(vector_to_return)
  device = None
  batch_sampler = BatchSampler(RandomSampler(dataset), batch_size, True)
  with monkeypatch.context() as m:
    m.setattr(nn, 'DataParallel', _.identity)
    m.setattr(u, 'tensors_to_device', lambda batch, device: batch)
    tester = t.Tester(dataset=dataset,
                      batch_sampler=batch_sampler,
                      model=model,
                      entity_embeds=entity_embeds,
                      embedding_lookup=embedding_lookup,
                      device=device)
    assert tester.test() == (torch.tensor(1), 1)