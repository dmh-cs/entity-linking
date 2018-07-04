import pydash as _
from functools import reduce
import torch
import torch.sparse as sparse
from parsers import parse_text_for_tokens, parse_for_tokens
import numpy as np


def _tokenize_page(page):
  return {'entity_name': page['title'],
          'tokens': parse_text_for_tokens(page['content'])}

def _tokens_to_embeddings(embedding_lookup, tokens):
  text_embeddings = []
  for token in tokens:
    if token in embedding_lookup:
      text_embeddings.append(embedding_lookup[token])
    elif token == 'MENTION_START_HERE':
      text_embeddings.append(embedding_lookup['<MENTION_START_HERE>'])
    elif token == 'MENTION_END_HERE':
      text_embeddings.append(embedding_lookup['<MENTION_END_HERE>'])
    else:
      text_embeddings.append(embedding_lookup['<UNK>'])
  return text_embeddings

def _tokens_to_padded_embeddings(embedding_lookup, tokens, batch_max_len=100) -> torch.Tensor:
  text_embeddings = _tokens_to_embeddings(embedding_lookup, tokens)
  if len(text_embeddings) < batch_max_len:
    text_embeddings.extend([embedding_lookup['<PAD>'] for _ in range(batch_max_len - len(text_embeddings))])
  return torch.stack(text_embeddings)

def _transform_raw_dataset(entity_lookup, embedding_lookup, raw_dataset):
  description_label_tuples = map(_.curry(transform_page, 3)(entity_lookup, embedding_lookup),
                                 raw_dataset)
  return map(list, zip(*description_label_tuples))

def transform_page(entity_lookup,
                   embedding_lookup,
                   page,
                   num_tokens=100,
                   use_entire_page=False):
  tokenized_page = _tokenize_page(page)
  if use_entire_page: raise NotImplementedError('Using entire pages is not yet implemented.')
  return (_tokens_to_padded_embeddings(embedding_lookup, tokenized_page['tokens'][:num_tokens]),
          entity_lookup[tokenized_page['entity_name']])

def transform_raw_datasets(entity_lookup, embedding_lookup, raw_datasets):
  return _.map_values(raw_datasets,
                      _.curry(_transform_raw_dataset)(entity_lookup, embedding_lookup))

def _find_mention_sentence_span(sentence_spans, mention_offset):
  return _.find(sentence_spans, lambda span: mention_offset >= span[0] and mention_offset <= span[1])

def _merge_sentences_across_mention(sentence_spans, mention_offset, mention_len):
  mention_end = mention_offset + mention_len
  span = _find_mention_sentence_span(sentence_spans, mention_offset)
  span_index = sentence_spans.index(span)
  while mention_end > span[1]:
    next_span = sentence_spans[span_index + 1]
    span = [span[0], next_span[1]]
  return span

def get_mention_sentence_splits(page_content, sentence_spans, mention_info):
  mention_len = len(mention_info['mention'])
  sentence_span = _merge_sentences_across_mention(sentence_spans, mention_info['offset'], mention_len)
  sentence = page_content[sentence_span[0] : sentence_span[1]]
  mention_index = sentence.index(mention_info['mention'])
  return [parse_for_tokens(sentence[:mention_index + mention_len]),
          parse_for_tokens(sentence[mention_index:])]

def _embed_sentence_splits(embedding_lookup, left_batch_len, right_batch_len, sentence_splits):
  flipped = reversed(sentence_splits[0])
  embedded_flipped_left = _tokens_to_padded_embeddings(embedding_lookup, flipped, left_batch_len)
  return [torch.tensor(np.flip(embedded_flipped_left.numpy(), 0).tolist()),
          _tokens_to_padded_embeddings(embedding_lookup, sentence_splits[1], right_batch_len)]

def _get_left_right_max_len(sentence_splits_batch):
  try:
    left_batch = [sentence_splits[0] for sentence_splits in sentence_splits_batch]
    right_batch = [sentence_splits[1] for sentence_splits in sentence_splits_batch]
  except:
    print(sentence_splits_batch)
    raise
  return max(map(len, left_batch)), max(map(len, right_batch))

def pad_and_embed_batch(embedding_lookup, sentence_splits_batch):
  embedded = []
  left_batch_len, right_batch_len = _get_left_right_max_len(sentence_splits_batch)
  for sentence_splits in sentence_splits_batch:
    embedded.append(_embed_sentence_splits(embedding_lookup,
                                           left_batch_len,
                                           right_batch_len,
                                           sentence_splits))
  return embedded

def _insert_mention_flags(page_content, mention_info):
  mention_text = mention_info['mention']
  start = mention_info['offset']
  end = mention_info['offset'] + len(mention_text)
  return page_content[:start] + 'MENTION_START_HERE ' + mention_text +  ' MENTION_END_HERE' + page_content[end:]

def embed_page_content(embedding_lookup, page_mention_infos, page_content):
  page_content_with_mention_flags = reduce(_insert_mention_flags,
                                           page_mention_infos,
                                           page_content)
  tokens = parse_text_for_tokens(page_content_with_mention_flags)
  return torch.stack(_tokens_to_embeddings(embedding_lookup, tokens))
