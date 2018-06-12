import torch

def build_cursor_generator(cursor, buff_len=1000):
  while True:
    results = cursor.fetchmany(buff_len)
    if not results: return
    for result in results: yield result

def get_batches(data, batch_size):
  args = [iter(data)] * batch_size
  for batch in zip(*args):
    yield torch.stack([torch.tensor(elem) for elem in batch])
