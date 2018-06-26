import pydash as _
from functools import reduce
import torch
import torch.sparse as sparse
from parsers import parse_text_for_tokens, parse_for_tokens


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

def _find_mention_sentence(sentences, mention_info):
  sentence_lengths = [len(sentence) for sentence in sentences]
  cumsum = reduce(lambda acc, length: [length] if _.is_empty(acc) else acc + [acc[-1] + length], sentence_lengths, [])
  index = _.find_index(cumsum, lambda ends_at: mention_info['offset'] <= ends_at)
  return sentences[index]

def get_mention_sentence_splits(sentences, mention_info):
  sentence = _find_mention_sentence(sentences, mention_info)
  mention_index = sentence.index(mention_info['mention'])
  mention_len = len(mention_info['mention'])
  return [parse_for_tokens(sentence[:mention_index + mention_len]),
          parse_for_tokens(sentence[mention_index:])]

def embed_sentence_splits(embedding_lookup, left_batch_len, right_batch_len, sample):
  embedded = [_tokens_to_embeddings(sample['sentence_splits'][0], embedding_lookup, left_batch_len),
              _tokens_to_embeddings(sample['sentence_splits'][1], embedding_lookup, right_batch_len)]
  return _.assign({}, sample, {'sentence_splits': embedded})
