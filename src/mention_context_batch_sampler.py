from random import shuffle
from torch.utils.data.sampler import Sampler
import pydash as _
import math
import random

class MentionContextBatchSampler(Sampler):
  def __init__(self, cursor, page_id_order, batch_size):
    super(MentionContextBatchSampler, self).__init__([])
    self.cursor = cursor
    self.page_id_order = page_id_order
    self.batch_size = batch_size
    self.page_ctr = 0
    self.ids_from_last_page = set()
    self.ids = []

  def __len__(self):
    raise NotImplementedError

  def __iter__(self):
    while self.page_ctr < len(self.page_id_order) or not _.is_empty(self.ids_from_last_page):
      yield self._get_next_batch()

  def _get_next_batch(self):
    ids = []
    if len(self.ids_from_last_page) > self.batch_size:
      ids = random.sample(list(self.ids_from_last_page), self.batch_size)
      self.ids_from_last_page = self.ids_from_last_page - set(ids)
      shuffle(ids)
      return ids
    else:
      if not _.is_empty(self.ids_from_last_page):
        ids = list(self.ids_from_last_page)
        self.ids_from_last_page = set()
        if self.page_ctr > len(self.page_id_order):
          return ids
      for page_id in self.page_id_order[self.page_ctr:]:
        self.page_ctr += 1
        self.cursor.execute('select id from mentions where page_id = %s', page_id)
        page_mention_ids = [row['id'] for row in self.cursor.fetchall()]
        ids.extend(page_mention_ids)
        if len(ids) >= self.batch_size:
          self.ids_from_last_page = set(ids[self.batch_size:])
          ids = ids[:self.batch_size]
          shuffle(ids)
          return ids
        else:
          self.ids_from_last_page = set()
      ids = ids[:]
      shuffle(ids)
      return ids
