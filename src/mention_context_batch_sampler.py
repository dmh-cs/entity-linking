from random import shuffle
from torch.utils.data.sampler import Sampler
import pydash as _
import math

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
    num_mentions = self._get_num_mentions()
    num_batches = math.ceil(num_mentions / self.batch_size)
    return num_batches

  def __iter__(self):
    while self.page_ctr < len(self.page_id_order):
      yield self._get_next_batch()

  def _get_next_batch(self):
    self.ids = []
    if len(self.ids_from_last_page) > self.batch_size:
      self.ids = list(self.ids_from_last_page)[:self.batch_size]
      self.ids_from_last_page = self.ids_from_last_page - set(self.ids)
      shuffle(self.ids)
      return self.ids
    else:
      if not _.is_empty(self.ids_from_last_page):
        self.page_ctr += 1
        self.ids = list(self.ids_from_last_page)
        self.ids_from_last_page = set()
    for page_id in self.page_id_order[self.page_ctr:]:
      self.cursor.execute('select id from mentions where page_id = %s', page_id)
      page_mention_ids = [row['id'] for row in self.cursor.fetchall()]
      self.ids.extend(page_mention_ids)
      if len(self.ids) >= self.batch_size:
        self.ids = self.ids[:self.batch_size]
        self.ids_from_last_page = set(page_mention_ids) - set(self.ids)
        shuffle(self.ids)
        return self.ids
      else:
        self.ids_from_last_page = set()
        self.page_ctr += 1
    self.ids = self.ids[:]
    shuffle(self.ids)
    return self.ids

  def _get_num_mentions(self):
    self.cursor.execute('select count(*) from mentions')
    return self.cursor.fetchone()['count(*)']
