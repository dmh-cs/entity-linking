import data_transformers as dt
import torch
import pydash as _

def test_transform_raw_datasets():
  entity_lookup = {'a': 1, 'b':2}
  embedding_lookup = {'verb': torch.tensor([1, 2, 3]),
                      'word': torch.tensor([2, 3, 4]),
                      '<PAD>': torch.tensor([0, 0, 0]),
                      '<UNK>': torch.tensor([1, 1, 1])}
  raw_datasets = {'train': [{'title': 'a', 'content': 'verb verb word verb'}]}
  datasets = dt.transform_raw_datasets(entity_lookup, embedding_lookup, raw_datasets)
  result = {'train': [(torch.tensor([[1, 2, 3],
                                     [1, 2, 3],
                                     [2, 3, 4],
                                     [1, 2, 3]] + [[0, 0, 0]] * 96),),
                      (1,)]}
  ctr = 0
  descriptions, label = datasets['train']
  for description, label in zip(descriptions, label):
    assert torch.equal(description, result['train'][0][0])
    assert label == result['train'][1][0]
    ctr += 1
  assert ctr == 1
