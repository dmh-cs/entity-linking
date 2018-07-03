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

def test_get_mention_sentence_splits():
  page_content = 'a b c aa bb cc'
  sentence_spans = [(0, 5), (6, 14)]
  mention_info = {'mention': 'bb cc', 'offset': 9}
  assert dt.get_mention_sentence_splits(page_content,
                                        sentence_spans,
                                        mention_info) == [['aa', 'bb', 'cc'], ['bb', 'cc']]

def test_get_mention_sentence_splits_with_merge():
  page_content = 'a b c aa bb cc'
  sentence_spans = [(0, 5), (6, 14)]
  mention_info = {'mention': 'c aa', 'offset': 4}
  assert dt.get_mention_sentence_splits(page_content,
                                        sentence_spans,
                                        mention_info) == [['a', 'b', 'c', 'aa'], ['c', 'aa', 'bb', 'cc']]


def test_pad_and_embed_batch():
  embedding_lookup = _.map_values({'<PAD>': [-1], '<UNK>': [0], 'a': [1], 'b': [2], 'c': [3], 'd': [4]},
                                  torch.tensor)
  sentence_splits = [[['a', 'b', 'c'], ['c', 'd']],
                     [['b', 'a'], ['a', 'd', 'd', '.']]]
  padded_and_embedded = [[torch.tensor([[1], [2], [3]]), torch.tensor([[3], [4], [-1], [-1]])],
                         [torch.tensor([[-1], [2], [1]]), torch.tensor([[1], [4], [4], [0]])]]
  result = dt.pad_and_embed_batch(embedding_lookup, sentence_splits)
  split_ctr = 0
  for sentence_num, sentence in enumerate(padded_and_embedded):
    for split_num, split in enumerate(sentence):
      assert torch.equal(split, result[sentence_num][split_num])
      split_ctr += 1
  assert split_ctr == 4
