import torch

import pydash as _

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
