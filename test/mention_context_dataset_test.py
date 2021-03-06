from mention_context_dataset import MentionContextDataset
import torch
import torch.nn as nn
from unittest.mock import Mock
from utils import coll_compare_keys_by
import pydash as _

def get_mock_cursor():
  cursor = Mock()
  cursor.execute = lambda q, args: None
  cursor.fetchone = lambda: {'count(*)': 5}
  return cursor

def compare_candidate_ids_tensor(expected, result):
  assert isinstance(result, torch.Tensor)
  expected_candidate_ids = set(expected.numpy())
  result_candidate_ids = set(result.numpy())
  assert _.is_empty(expected_candidate_ids - result_candidate_ids)
  for generated_candidate in result_candidate_ids - expected_candidate_ids:
    assert generated_candidate not in expected_candidate_ids
  return True

def test_mention_context_dataset():
  cursor = get_mock_cursor()
  page_id_order = [3, 1, 2]
  batch_size = 5
  entity_candidates_prior = {'aa': {1: 20},
                             'bb': {0: 10, 1: 2},
                             'cc': {2: 3}}
  entity_label_lookup = dict(zip(range(5), range(5)))
  embedding_dim = 1
  embedding_dict = {'<PAD>': torch.tensor([0]),
                    '<UNK>': torch.tensor([2]),
                    '<MENTION_START_HERE>': torch.tensor([-1]),
                    '<MENTION_END_HERE>': torch.tensor([-2])}
  token_idx_lookup = dict(zip(embedding_dict.keys(),
                              range(len(embedding_dict))))
  embedding = nn.Embedding.from_pretrained(torch.stack([embedding_dict[token] for token in token_idx_lookup]))
  num_entities = 5
  num_candidates = 2
  dataset = MentionContextDataset(cursor,
                                  page_id_order,
                                  entity_candidates_prior,
                                  entity_label_lookup,
                                  embedding,
                                  token_idx_lookup,
                                  batch_size,
                                  num_entities,
                                  num_candidates)
  dataset._mention_infos = {0: {'mention': 'bb', 'offset': 9, 'page_id': 2, 'entity_id': 0, 'mention_id': 0},
                            1: {'mention': 'aa', 'offset': 6, 'page_id': 2, 'entity_id': 1, 'mention_id': 1},
                            2: {'mention': 'cc', 'offset': 0, 'page_id': 1, 'entity_id': 2, 'mention_id': 2},
                            3: {'mention': 'bb', 'offset': 3, 'page_id': 1, 'entity_id': 0, 'mention_id': 3},
                            4: {'mention': 'bb', 'offset': 3, 'page_id': 0, 'entity_id': 1, 'mention_id': 4}}
  dataset._page_content_lookup = {2: 'a b c aa bb',
                                  1: 'cc bb c b a',
                                  0: 'dd bb a b c'}
  dataset._sentence_spans_lookup = {2: [(0, 5), (6, 11)],
                                    1: [(0, 5), (6, 11)],
                                    0: [(0, 5), (6, 11)]}
  dataset._embedded_page_content_lookup = {2: [0, 1],
                                           1: [1, 2],
                                           0: [1]}
  dataset._entity_page_mentions_lookup = {2: [[0]],
                                          1: [[1]],
                                          0: [[1]]}
  dataset._mentions_per_page_ctr = {2: 2,
                                    1: 2,
                                    0: 1}
  expected_data = [{'sentence_splits': [['aa', 'bb'], ['bb']],
                    'label': 0,
                    'embedded_page_content': [0, 1],
                    'entity_page_mentions': [[0]],
                    'candidate_ids': torch.tensor([0, 1]),
                    'p_prior': torch.tensor([10/12, 2/12])},
                   {'sentence_splits': [['aa'], ['aa', 'bb']],
                    'label': 1,
                    'embedded_page_content': [0, 1],
                    'entity_page_mentions': [[0]],
                    'candidate_ids': torch.tensor([1]),
                    'p_prior': torch.tensor([1.0])},
                   {'sentence_splits': [['cc'], ['cc', 'bb']],
                    'label': 2,
                    'embedded_page_content': [1, 2],
                    'entity_page_mentions': [[1]],
                    'candidate_ids': torch.tensor([2]),
                    'p_prior': torch.tensor([1.0])},
                   {'sentence_splits': [['cc', 'bb'], ['bb']],
                    'label': 0,
                    'embedded_page_content': [1, 2],
                    'entity_page_mentions': [[1]],
                    'candidate_ids': torch.tensor([0, 1]),
                    'p_prior': torch.tensor([10/12, 2/12])},
                   {'sentence_splits': [['dd', 'bb'], ['bb']],
                    'label': 1,
                    'embedded_page_content': [1],
                    'entity_page_mentions': [[1]],
                    'candidate_ids': torch.tensor([0, 1]),
                    'p_prior': torch.tensor([10/12, 2/12])}]
  iterator = iter(dataset)
  dataset_values = [next(iterator) for _ in range(len(expected_data))]
  comparison = {'sentence_splits': _.is_equal,
                'label': _.is_equal,
                'embedded_page_content': _.is_equal,
                'entity_page_mentions': _.is_equal,
                'candidate_ids': compare_candidate_ids_tensor,
                'p_prior': lambda a, b: len(a) == len(_.intersection(a.tolist(), b.tolist()))}
  assert coll_compare_keys_by(expected_data, dataset_values, comparison)
