import pydash as _
from functools import reduce
import torch
import torch.sparse as sparse
from parsers import parse_text_for_tokens, parse_for_tokens
import numpy as np


def _tokenize_page(page):
  return {'entity_name': page['title'],
          'tokens': parse_text_for_tokens(page['content'])}

def _tokens_to_embeddings(tokens, embedding_lookup, desc_len=100) -> torch.Tensor:
  desc_vec = []
  for token in tokens:
    if token in embedding_lookup:
      desc_vec.append(embedding_lookup[token])
    else:
      desc_vec.append(embedding_lookup['<UNK>'])
  if len(desc_vec) < desc_len:
    desc_vec.extend([embedding_lookup['<PAD>'] for _ in range(desc_len - len(desc_vec))])
  return torch.stack(desc_vec)

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
  return (_tokens_to_embeddings(tokenized_page['tokens'][:num_tokens], embedding_lookup),
          entity_lookup[tokenized_page['entity_name']])

def transform_raw_datasets(entity_lookup, embedding_lookup, raw_datasets):
  return _.map_values(raw_datasets,
                      _.curry(_transform_raw_dataset)(entity_lookup, embedding_lookup))

def _find_mention_sentence_index(sentences, mention_info):
  sentence_lengths = [len(sentence) for sentence in sentences]
  def _reducer(acc, length__sentence_num):
    sentence_num = length__sentence_num[0]
    length = length__sentence_num[1]
    if sentence_num == 0:
      return [length]
    else:
      return acc + [acc[-1] + length + 1]
  cumsum = reduce(_reducer, enumerate(sentence_lengths), [])
  index = _.find_index(cumsum, lambda ends_at: mention_info['offset'] <= ends_at)
  return index

def get_mention_sentence_splits(sentences, mention_info):
  try:
    sentence_index = _find_mention_sentence_index(sentences, mention_info)
    indices_to_check = [sentence_index, sentence_index - 1, sentence_index + 1]
    for index in indices_to_check:
      try:
        sentence = sentences[index]
        mention_index = sentence.index(mention_info['mention'])
        break
      except ValueError:
        print('Mention not found in sentence', mention_info, sentence)
      except IndexError:
        continue
    mention_len = len(mention_info['mention'])
    return [parse_for_tokens(sentence[:mention_index + mention_len]),
            parse_for_tokens(sentence[mention_index:])]
  except ValueError:
    print('Mention sentence not found', mention_info)
    raise

def _embed_sentence_splits(embedding_lookup, left_batch_len, right_batch_len, sentence_splits):
  flipped = reversed(sentence_splits[0])
  embedded_flipped_left = _tokens_to_embeddings(flipped, embedding_lookup, left_batch_len)
  return [torch.tensor(np.flip(embedded_flipped_left.numpy(), 0).tolist()),
          _tokens_to_embeddings(sentence_splits[1], embedding_lookup, right_batch_len)]

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
