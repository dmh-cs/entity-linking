import data_transformers as dt
import torch
import torch.nn as nn
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

def test_embed_page_content():
  embedding_lookup = _.map_values({'<PAD>': [-1],
                                   '<UNK>': [0],
                                   'MENTION_START_HERE': [-2], 'MENTION_END_HERE': [-3],
                                   'a': [1], 'b': [2], 'c': [3], 'd': [4]},
                                  torch.tensor)
  page_mention_infos = [{'offset': 2, 'mention': 'b c'}]
  page_content = 'a b c d'
  embedded = torch.tensor([[1], [-2], [2], [3], [-3], [4]])
  assert torch.equal(dt.embed_page_content(embedding_lookup, page_content, page_mention_infos), embedded)

def test_pad_batch():
  pad_vector = torch.tensor([0])
  batch = [torch.tensor(vec) for vec in [[[1]], [[1], [2]]]]
  min_len = 3
  assert torch.equal(dt.pad_batch(pad_vector, batch, min_len=min_len),
                     torch.tensor([[[1], [0], [0]],
                                   [[1], [2], [0]]]))

def test_pad_batch_no_min():
  pad_vector = torch.tensor([0])
  batch = [torch.tensor(vec) for vec in [[[1]], [[1], [2]]]]
  assert torch.equal(dt.pad_batch(pad_vector, batch),
                     torch.tensor([[[1], [0]],
                                   [[1], [2]]]))

def test_embed_and_pack_batch():
  embedding_lookup = {'a': torch.tensor([1]), 'b': torch.tensor([2])}
  sentence_splits_batch = [[['a', 'b', 'a', 'b'], ['b', 'a']],
                           [['b', 'a'], ['a', 'b', 'a', 'b']]]
  left = [torch.tensor(vec) for vec in [[[1], [2], [1], [2]], [[2], [1]]]]
  right = [torch.tensor(vec) for vec in [[[1], [2], [1], [2]], [[2], [1]]]]
  result = dt.embed_and_pack_batch(embedding_lookup, sentence_splits_batch)
  assert torch.equal(result[0]['embeddings'].data,
                     nn.utils.rnn.pack_sequence(left).data)
  assert torch.equal(result[0]['embeddings'].batch_sizes,
                     nn.utils.rnn.pack_sequence(left).batch_sizes)
  assert result[0]['order'] == [0, 1]
  assert torch.equal(result[0]['embeddings'].data,
                     nn.utils.rnn.pack_sequence(right).data)
  assert torch.equal(result[0]['embeddings'].batch_sizes,
                     nn.utils.rnn.pack_sequence(right).batch_sizes)
  assert result[1]['order'] == [1, 0]

def test__find_mention_sentence_span():
  sentence_spans = [(0, 3), (4, 8), (8, 12), (13, 19)]
  mention_offset = 10
  span = dt._find_mention_sentence_span(sentence_spans, mention_offset)
  assert span == (8, 12)
