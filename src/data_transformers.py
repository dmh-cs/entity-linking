import pydash as _
from functools import reduce
import torch
from parsers import parse_text_for_tokens


def _tokenize_page(page):
  return {'entity_name': page['title'],
          'tokens': parse_text_for_tokens(page['content'])}

def _build_description(tokens, embedding_lookup, desc_len=100) -> torch.Tensor:
  desc_vec = []
  for token in tokens:
    if token in embedding_lookup:
      desc_vec.append(embedding_lookup[token])
    else:
      desc_vec.append(embedding_lookup['<UNK>'])
  if len(desc_vec) < desc_len:
    desc_vec.extend([embedding_lookup['<PAD>'] for _ in range(desc_len - len(desc_vec))])
  return torch.stack(desc_vec)

def _transform_page(entity_lookup,
                    embedding_lookup,
                    page,
                    num_tokens=100,
                    use_entire_page=False):
  tokenized_page = _tokenize_page(page)
  if use_entire_page: raise NotImplementedError('Using entire pages is not yet implemented.')
  return (_build_description(tokenized_page['tokens'][:num_tokens], embedding_lookup),
          entity_lookup[tokenized_page['entity_name']])

def _transform_raw_dataset(entity_lookup, embedding_lookup, raw_dataset):
  description_label_tuples = map(_.curry(_transform_page, 3)(entity_lookup, embedding_lookup),
                                 raw_dataset)
  descriptions = map(lambda tup: tup[0], description_label_tuples)
  labels = map(lambda tup: tup[1], description_label_tuples)
  return descriptions, labels

def transform_raw_datasets(entity_lookup, embedding_lookup, raw_datasets):
  return _.map_values(raw_datasets,
                      _.curry(_transform_raw_dataset)(entity_lookup, embedding_lookup))
