from torch.utils.data.sampler import Sampler, SubsetRandomSampler

class FixLenSequentialSampler(Sampler):
  def __init__(self, length):
    self.length = length

  def __iter__(self):
    return iter(range(self.length))

  def __len__(self):
    return self.length

class SubsetSequentialSampler(Sampler):
  def __init__(self, indices):
    self.indices = indices

  def __iter__(self):
    return (self.indices[i] for i in range(len(self.indices)))

  def __len__(self):
    return len(self.indices)

class _ChunkedIter(object):
  def __init__(self, _len, chunk_size):
    self.len = _len
    self.chunk_size = chunk_size
    self.index = 0
    self.chunk_iter = None

  def __next__(self):
    if (self.index % self.chunk_size) == 0:
      self.chunk_iter = iter(SubsetRandomSampler(range(self.index,
                                                       min(self.index + self.chunk_size,
                                                           self.len))))
    result = next(self.chunk_iter)
    self.index += 1
    return result

class ChunkedRandomSampler(Sampler):
  """
  Random sampler that operates efficiently in conjunction with
  ChunkedDataset
  """
  def __init__(self, len_, super_batch_size):
    """

    :param Dataset dataset: Dataset being sampled from
    :param int super_batch_size: super-batch size to chunk with
    """
    super(ChunkedRandomSampler, self)
    self.chunk_size = super_batch_size
    self.len = len_

  def __len__(self):
    return self.len

  def __iter__(self):
    return _ChunkedIter(self.len, self.chunk_size)
