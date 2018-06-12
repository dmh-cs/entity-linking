import data_transformers as dt
import torch
import pydash as _

def test_transform_raw_datasets():
  entity_lookup = {'a': 1, 'b':2}
  embedding_lookup = {'verb': [1, 2, 3], 'word': [2, 3, 4]}
  raw_datasets = {'train': [{'title': 'a', 'content': 'verb verb word verb'}]}
  datasets = dt.transform_raw_datasets(entity_lookup, embedding_lookup, raw_datasets)
  datasets['train'] = list(datasets['train'])
  result = {'train': [(torch.tensor([[1, 2, 3],
                                     [1, 2, 3],
                                     [2, 3, 4],
                                     [1, 2, 3]]),
                       1)]}
  assert torch.equal(datasets['train'][0][0], result['train'][0][0])
  assert datasets['train'][0][1] == result['train'][0][1]
  assert len(datasets['train']) == len(result['train'])
