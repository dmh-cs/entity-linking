import json
import pickle

def write_to_file(path, rows):
  if '.pkl' in path:
    with open(path, 'wb+') as fh:
      pickle.dump(rows, fh)
  else:
    with open(path, 'w+') as fh:
      json.dump(rows, fh)

def read_from_file(path):
  if '.pkl' in path:
    with open(path, 'rb') as fh:
      return pickle.load(fh)
  else:
    with open(path, 'r') as fh:
      return json.load(fh)

def read_cache(path, fn):
  path = './caches/' + path
  try:
    data = read_from_file(path)
  except FileNotFoundError:
    data = fn()
    write_to_file(path, data)
  return data
