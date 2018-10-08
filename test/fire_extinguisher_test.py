from torch.utils.data.sampler import BatchSampler, SequentialSampler

from fire_extinguisher import BatchRepeater

def test_batch_repeater():
  data = list(range(10))
  batch_size = 5
  batch_sampler = BatchSampler(SequentialSampler(data), batch_size, False)
  batch_repeater = BatchRepeater(batch_sampler, num_repetitions=3)
  batches = []
  for batch in batch_repeater:
    batches.append(batch)
  assert all([set(batch) == set([0, 1, 2, 3, 4]) for batch in batches])
  assert len(batches) == 3
