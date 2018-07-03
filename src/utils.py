import torch
import pydash

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
