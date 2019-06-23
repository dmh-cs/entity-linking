import torch
import torch.nn as nn
from collections import defaultdict
import unidecode
from itertools import product
from random import shuffle

import pydash as _
from pyrsistent import freeze, thaw

def to_idx(token_idx_lookup, token):
  if token in token_idx_lookup:
    return token_idx_lookup[token]
  elif token.lower() in token_idx_lookup:
    return token_idx_lookup[token.lower()]
  else:
    return token_idx_lookup['<UNK>']

def build_cursor_generator(cursor, buff_len=1000):
  while True:
    results = cursor.fetchmany(buff_len)
    if not results: return
    for result in results: yield result

def get_batches(data, batch_size):
  args = [iter(data)] * batch_size
  for batch in zip(*args):
    yield torch.stack([torch.tensor(elem) for elem in batch])

def compare_keys_by(obj1, obj2, comp_with):
  assert set(obj1.keys()) == set(obj2.keys())
  for key, obj1_val in obj1.items():
    obj2_val = obj2[key]
    comparison = comp_with[key]
    assert comparison(obj1_val, obj2_val)
  return True

def coll_compare_keys_by(coll1, coll2, comp_with):
  assert len(coll1) == len(coll2)
  for elem1, elem2 in zip(coll1, coll2):
    assert compare_keys_by(elem1, elem2, comp_with)
  return True

def tensors_to_device(obj, device):
  return {key: val.to(device) if isinstance(val, torch.Tensor) else val for key, val in obj.items()}

def sort_index(coll, key=_.identity, reverse=False):
  return [index for index, _ in sorted(enumerate(coll),
                                       key=lambda elem: key(elem[1]),
                                       reverse=reverse)]

def append_create(obj, key, val):
  if key in obj:
    obj[key].append(val)
  else:
    obj[key] = [val]

def chunk_apply(fn, coll, chunk_size):
  start = 0
  result = []
  while start < len(coll):
    result.extend(fn(coll[start : start + chunk_size]))
    start += chunk_size
  return result

def chunk_apply_at_lim(fn, coll, lim):
  if len(coll) > lim:
    return chunk_apply(fn, coll, lim)
  else:
    return fn(coll)

def get_prior_approx_mapping(entity_candidates_prior):
  approx_mapping = defaultdict(list)
  for mention in entity_candidates_prior.keys():
    approx_mention = unidecode.unidecode(mention).lower()
    approx_mapping[approx_mention].append(mention)
  return approx_mapping

class Identity(nn.Module):
  def forward(self, x): return x

def items_to_str(items, sort_by=None):
  if sort_by is not None:
    to_serialize = sorted(items, key=sort_by)
  else:
    to_serialize = items
  return '_'.join(':'.join(str(elem) for elem in pair)
                  for pair in to_serialize)

def hparam_search(p, arg_options, rand_p=False):
  paths = [options_details['path'] for options_details in arg_options]
  options_grid = product(*[options_details['options']
                           for options_details in arg_options])
  hparams = [thaw(p)]
  new_options = [[]]
  for options in options_grid:
    new_params = thaw(p)
    new_options.append(list(zip(paths, options)))
    for path, option in zip(paths, options):
      _.set_(new_params, path, option)
    hparams.append(new_params)
  filtered = []
  for cand_p, opts in zip(hparams, new_options):
    if all(options_details['if'](cand_p) or (_.get(thaw(cand_p), options_details['path']) == options_details['options'][0])
           for options_details in arg_options if 'if' in options_details):
      filtered.append((cand_p, opts))
  if rand_p: shuffle(filtered)
  return freeze(filtered)
