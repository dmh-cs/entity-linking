from unittest.mock import Mock
import pytest
import string

import pydash as _
import torch
from torch.utils.data.sampler import BatchSampler, RandomSampler

import torch.nn as nn
import utils as u

import tester as t


@pytest.fixture()
def myMock():
  mock_call = Mock.__call__
  yield Mock
  Mock.__call__ = mock_call

def get_mock_model(vector_to_return):
  model = Mock()
  model.to = lambda device: model
  Mock.__call__ = lambda self, x: torch.unsqueeze(vector_to_return, 0)
  return model

def test_tester(monkeypatch, myMock):
  dataset = [{'label': 0,
              'sentence_splits': [['a', 'b', 'c'], ['c', 'd']],
              'candidates': torch.tensor([0, 1]),
              'embedded_page_content': torch.tensor([[1], [-2], [2], [3], [-3], [4]])},
             {'label': 2,
              'sentence_splits': [['a', 'b', 'c'], ['c', 'd']],
              'candidates': torch.tensor([2, 1]),
              'embedded_page_content': torch.tensor([[1], [-2], [2], [3], [-3], [4]])},
             {'label': 1,
              'sentence_splits': [['a', 'b', 'c'], ['c', 'd']],
              'candidates': torch.tensor([3, 1]),
              'embedded_page_content': torch.tensor([[1], [-2], [2], [3], [-3], [4]])}]
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
    labels_for_batch = tester._get_labels_for_batch(torch.tensor([elem['label'] for elem in dataset]),
                                                    torch.tensor([[1, 0], [4, 5], [1, 0]]))
    assert torch.equal(labels_for_batch,
                       torch.tensor([1, -1, 0]))
