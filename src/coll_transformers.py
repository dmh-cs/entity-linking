class KeyMapper():
  def __init__(self, coll, fn):
    self.fn = fn
    self.coll = coll

  def __getitem__(self, idx): return self.coll[self.fn(idx)]

class KeyMapperByLookup(KeyMapper):
  def __init__(self, coll, lookup):
    super().__init__(coll, lambda idx: lookup[idx])

class ValuesMapper():
  def __init__(self, coll, fn):
    self.fn = fn
    self.coll = coll

  def __getitem__(self, idx):
    if hasattr(idx, '__iter__'):
      return [self.fn(val) if val is not None else None for val in self.coll[idx]]
    else:
      return self.fn(self.coll[idx])

class DefaultVal():
  def __init__(self, coll, default_value):
    self.default_value = default_value
    self.coll = coll

  def __getitem__(self, idx):
    if hasattr(idx, '__iter__'):
      return [elem if elem is not None else self.default_value
              for elem in self.coll[idx]]
    else:
      try:               return self.coll[idx]
      except IndexError: return self.default_value
