import torch
import pydash as _

import data_fetchers as df

def test_get_random_indexes():
  result = df.get_random_indexes(300, [2, 8], 298)
  assert 2 not in result
  assert 8 not in result
  assert len(result) == 298
  assert len(set(result)) == 298

def test_get_random_indexes_too_long():
  caught = False
  try:
    df.get_random_indexes(10, [2, 8], 12)
  except Exception as e:
    assert isinstance(e, ValueError)
    caught = True
  assert caught

def test_get_candidate_ids():
  entity_candidates_prior = {'a': {1: 20}, 'b': {2: 12}, 'c': {3: 3}}
  num_entities = 300
  num_candidates = 300
  mention = 'b'
  label = 2
  candidate_ids = df.get_candidate_ids(entity_candidates_prior,
                                 num_entities,
                                 num_candidates,
                                 mention,
                                 label)
  assert 2 in candidate_ids.tolist()
  assert len(candidate_ids) == num_candidates
  assert len(set(candidate_ids.tolist())) == num_candidates
