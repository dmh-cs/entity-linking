from functools import reduce

import pydash as _
import torch
import torch.nn as nn

from utils import sort_index
from parsers import parse_text_for_tokens, parse_for_tokens


def pad_batch(pad_vector, batch, min_len=0):
  pad_to_len = max(min_len, max(_.map_(batch, len)))
  to_stack = []
  for elem in batch:
    dim_len = len(elem)
    if pad_to_len != dim_len:
      pad = torch.stack([pad_vector] * (pad_to_len - dim_len))
      to_stack.append(torch.cat((elem, pad), 0))
    else:
      to_stack.append(elem)
  return torch.stack(to_stack)

def pad_batch_list(pad_elem, batch, min_len=0):
  assert isinstance(batch, list)
  assert isinstance(pad_elem, (int, str))
  pad_to_len = max(min_len, max(_.map_(batch, len)))
  to_stack = []
  for elem in batch:
    dim_len = len(elem)
    if pad_to_len != dim_len:
      pad = [pad_elem for i in range(pad_to_len - dim_len)]
      to_stack.append(elem + pad)
    else:
      to_stack.append(elem)
  return torch.stack(to_stack)

def _tokens_to_embeddings(embedding, token_idx_lookup, tokens):
  text_idxs = []
  for token in tokens:
    if token in token_idx_lookup:
      text_idxs.append(token_idx_lookup[token])
    elif token.lower() in token_idx_lookup:
      text_idxs.append(token_idx_lookup[token.lower()])
    elif token == 'MENTION_START_HERE':
      text_idxs.append(token_idx_lookup['<MENTION_START_HERE>'])
    elif token == 'MENTION_END_HERE':
      text_idxs.append(token_idx_lookup['<MENTION_END_HERE>'])
    else:
      text_idxs.append(token_idx_lookup['<UNK>'])
  return embedding(torch.tensor(text_idxs,
                                dtype=torch.long,
                                device=embedding.weight.device))

def _satisfies(span, offset):
  return offset >= span[0] and offset <= span[1]

def _search(spans, offset):
  if len(spans) == 1 and not _satisfies(spans[0], offset):
    return None
  index = int(len(spans) / 2)
  if _satisfies(spans[index], offset):
    return spans[index]
  else:
    return _search(spans[:index], offset) or _search(spans[index:], offset)

def _find_mention_sentence_span(sentence_spans, mention_offset):
  return _search(sentence_spans, mention_offset)

def _merge_sentences_across_mention(sentence_spans, mention_offset, mention_len):
  mention_end = mention_offset + mention_len
  span = _find_mention_sentence_span(sentence_spans, mention_offset)
  span_index = sentence_spans.index(span)
  end_offset = span[1]
  while mention_end > end_offset:
    span_index += 1
    next_span = sentence_spans[span_index]
    end_offset = next_span[1]
  span = [span[0], end_offset]
  return span

def get_mention_sentence_splits(page_content, sentence_spans, mention_info):
  mention_len = len(mention_info['mention'])
  sentence_span = _merge_sentences_across_mention(sentence_spans, mention_info['offset'], mention_len)
  sentence = page_content[sentence_span[0] : sentence_span[1]]
  mention_index = sentence.index(mention_info['mention'])
  return [parse_for_tokens(sentence[:mention_index + mention_len]),
          parse_for_tokens(sentence[mention_index:])]

def get_splits_and_order(packed):
  return packed['embeddings'], packed['order']

def embed_and_pack_batch(embedding, token_idx_lookup, sentence_splits_batch):
  left_order = sort_index(sentence_splits_batch, key=lambda split: len(split[0]), reverse=True)
  right_order = sort_index(sentence_splits_batch, key=lambda split: len(split[1]), reverse=True)
  left_batch = []
  right_batch = []
  for left_index, right_index in zip(left_order, right_order):
    split_left = sentence_splits_batch[left_index]
    split_right = sentence_splits_batch[right_index]
    left_batch.append(_tokens_to_embeddings(embedding,
                                            token_idx_lookup,
                                            split_left[0]))
    right_batch.append(_tokens_to_embeddings(embedding,
                                             token_idx_lookup,
                                             split_right[1]))
  left_unsort_order = sort_index(left_order)
  right_unsort_order = sort_index(right_order)
  return ({'embeddings': nn.utils.rnn.pack_sequence(left_batch), 'order': left_unsort_order},
          {'embeddings': nn.utils.rnn.pack_sequence(right_batch), 'order': right_unsort_order})

def _insert_mention_flags(page_content, mention_info):
  mention_text = mention_info['mention']
  start = mention_info['offset']
  end = mention_info['offset'] + len(mention_text)
  return page_content[:start] + 'MENTION_START_HERE ' + mention_text +  ' MENTION_END_HERE' + page_content[end:]

def embed_page_content(embedding, token_idx_lookup, page_content, page_mention_infos=[]):
  page_content_with_mention_flags = reduce(_insert_mention_flags,
                                           page_mention_infos,
                                           page_content)
  tokens = parse_text_for_tokens(page_content_with_mention_flags)
  return _tokens_to_embeddings(embedding, token_idx_lookup, tokens)
