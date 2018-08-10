from unittest.mock import Mock, create_autospec
import pytest
import string

import pydash as _
import torch
import torch.nn as nn
from torch.utils.data.sampler import BatchSampler, RandomSampler

import utils as u
from experiment import Experiment
from logits import Logits
from softmax import Softmax

import tester as t


@pytest.fixture()
def myMock():
  mock_call = Mock.__call__
  yield Mock
  Mock.__call__ = mock_call

def get_mock_model(vector_to_return):
  model = Mock()
  model.to = lambda device: model
  Mock.__call__ = lambda self, x: (None, vector_to_return)
  return model

def test_tester(monkeypatch, myMock):
  dataset = [{'label': 0,
              'sentence_splits': [['a', 'b', 'c'], ['c', 'd']],
              'candidates': torch.tensor([0, 1]),
              'embedded_page_content': torch.tensor([[1], [-2], [2], [3], [-3], [4]]),
              'entity_page_mentions': torch.tensor([[1], [-2], [0], [3], [0], [4]]),
              'p_prior': torch.tensor([0.1, 0.9])},
             {'label': 2,
              'sentence_splits': [['a', 'b', 'c'], ['c', 'd']],
              'candidates': torch.tensor([2, 1]),
              'embedded_page_content': torch.tensor([[1], [-2], [2], [3], [-3], [4]]),
              'entity_page_mentions': torch.tensor([[1], [-2], [0], [3], [0], [4]]),
              'p_prior': torch.tensor([0.1, 0.9])},
             {'label': 1,
              'sentence_splits': [['a', 'b', 'c'], ['c', 'd']],
              'candidates': torch.tensor([3, 1]),
              'embedded_page_content': torch.tensor([[1], [-2], [2], [3], [-3], [4]]),
              'entity_page_mentions': torch.tensor([[1], [-2], [0], [3], [0], [4]]),
              'p_prior': torch.tensor([0.1, 0.9])}]
  num_entities = 10
  embed_len = 200
  batch_size = 3
  entity_embeds = nn.Embedding(num_entities,
                               embed_len,
                               _weight=torch.randn((num_entities, embed_len)))
  embedding_lookup = dict(zip(string.ascii_lowercase,
                              [torch.tensor(i) for i, char in enumerate(string.ascii_lowercase)]))
  vector_to_return = entity_embeds(torch.tensor([1, 1, 1]))
  model = get_mock_model(vector_to_return)
  device = None
  batch_sampler = BatchSampler(RandomSampler(dataset), batch_size, True)
  mock_experiment = create_autospec(Experiment, instance=True)
  calc_logits = Logits()
  softmax = Softmax()
  logits_and_softmax = {'mention': lambda hidden, candidates_or_targets: softmax(calc_logits(hidden,
                                                                                             candidates_or_targets))}
  with monkeypatch.context() as m:
    m.setattr(nn, 'DataParallel', _.identity)
    m.setattr(u, 'tensors_to_device', lambda batch, device: batch)
    tester = t.Tester(dataset=dataset,
                      batch_sampler=batch_sampler,
                      model=model,
                      logits_and_softmax=logits_and_softmax,
                      entity_embeds=entity_embeds,
                      embedding_lookup=embedding_lookup,
                      device=device,
                      experiment=mock_experiment,
                      ablation=['prior', 'local_context', 'document_context'])
    assert tester.test() == (1, 3)
    labels_for_batch = tester._get_labels_for_batch(torch.tensor([elem['label'] for elem in dataset]),
                                                    torch.tensor([[1, 0], [4, 5], [1, 0]]))
    assert torch.equal(labels_for_batch,
                       torch.tensor([1, -1, 0]))
