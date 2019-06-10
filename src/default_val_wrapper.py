class DefaultVal():
  def __init__(self, coll, default_value):
    self.default_value = default_value
    self.coll = coll

  def __getitem__(self, idx):
    try:
      return self.coll[idx]
    except IndexError:
      return self.default_value
