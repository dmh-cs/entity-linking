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

  def __getitem__(self, idx): return self.fn(self.coll[idx])

class DefaultVal():
  def __init__(self, coll, default_value):
    self.default_value = default_value
    self.coll = coll

  def __getitem__(self, idx):
    try:
      return self.coll[idx]
    except IndexError:
      return self.default_value
