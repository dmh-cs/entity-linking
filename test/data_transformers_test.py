import data_transformers as dt
import torch
import pydash as _

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


def test_embed_page_content():
  embedding_lookup = _.map_values({'<PAD>': [-1],
                                   '<UNK>': [0],
                                   'MENTION_START_HERE': [-2], 'MENTION_END_HERE': [-3],
                                   'a': [1], 'b': [2], 'c': [3], 'd': [4]},
                                  torch.tensor)
  page_mention_infos = [{'offset': 2, 'mention': 'b c'}]
  page_content = 'a b c d'
  embedded = torch.tensor([[1], [-2], [2], [3], [-3], [4]])
  assert torch.equal(dt.embed_page_content(embedding_lookup, page_mention_infos, page_content), embedded)
