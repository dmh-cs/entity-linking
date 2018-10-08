from random import shuffle

class BatchRepeater():
  def __init__(self, sampler, num_repetitions=10000, repeat_first=True):
    self.batch_to_repeat = None
    self.sampler = iter(sampler)
    self.num_repetitions = num_repetitions
    self.repetition_ctr = 0
    if not repeat_first:
      raise NotImplementedError

  def __len__(self):
    return self.num_repetitions

  def __iter__(self):
    if self.batch_to_repeat is None:
      self.batch_to_repeat = next(self.sampler)
    while self.repetition_ctr < self.num_repetitions:
      shuffle(self.batch_to_repeat)
      self.repetition_ctr += 1
      yield self.batch_to_repeat
