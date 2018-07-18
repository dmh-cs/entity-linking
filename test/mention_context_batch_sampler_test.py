from mention_context_batch_sampler import MentionContextBatchSampler
from unittest.mock import Mock
import math
import pydash as _

def get_mock_cursor(mentions_by_page):
  cursor = Mock()
  cursor._data = {'mentions': mentions_by_page}
  cursor.execute = lambda query, page_id: cursor._data.update({'page_id': page_id})
  cursor.fetchall = lambda: [{'id': _id} for _id in cursor._data['mentions'][cursor._data['page_id']]]
  return cursor

def test_mention_context_batch_sampler():
  mentions_by_page = {0: [40, 50, 60, 70, 80, 90], 1: [100], 2: [0, 10, 20, 30]}
  cursor = get_mock_cursor(mentions_by_page)
  batch_size = 5
  page_id_order = [2, 0, 1]
  mentions_in_page_order = _.mapcat(page_id_order, lambda page_id: mentions_by_page[page_id])
  batch_sampler = MentionContextBatchSampler(cursor, page_id_order, batch_size, len(mentions_in_page_order))
  batches_seen = []
  indexes_seen = []
  for batch_num, batch_indexes in enumerate(batch_sampler):
    assert _.is_empty(_.intersection(batch_indexes, indexes_seen))
    indexes_seen.extend(batch_indexes)
    if batch_num == 0:
      assert len(set(batch_indexes) - {0, 10, 20, 30}) == 1
      assert any([mention in set(batch_indexes) - {0, 10, 20, 30} for mention in mentions_by_page[0]])
      assert len(batch_indexes) == batch_size
    elif batch_num == 1:
      assert _.is_empty(set(batch_indexes) - set(mentions_by_page[0]))
      assert len(batch_indexes) == batch_size
    elif batch_num == 2:
      assert batch_indexes == [100]
      assert len(batch_indexes) == 1
    batches_seen.append(batch_num)
  assert _.is_empty(_.difference(mentions_in_page_order, indexes_seen))
  assert batches_seen == [0, 1, 2]

def test_mention_context_batch_sampler_many_last_ids():
  mentions_by_page = {0: [120], 1: [130], 2: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]}
  cursor = get_mock_cursor(mentions_by_page)
  batch_size = 5
  page_id_order = [2, 0, 1]
  mentions_in_page_order = _.mapcat(page_id_order, lambda page_id: mentions_by_page[page_id])
  batch_sampler = MentionContextBatchSampler(cursor, page_id_order, batch_size, len(mentions_in_page_order))
  batches_seen = []
  indexes_seen = []
  for batch_num, batch_indexes in enumerate(batch_sampler):
    assert _.is_empty(_.intersection(batch_indexes, indexes_seen))
    indexes_seen.extend(batch_indexes)
    if batch_num == 0:
      assert _.is_empty(_.difference(batch_indexes, mentions_by_page[2]))
      assert len(batch_indexes) == batch_size
    elif batch_num == 1:
      assert _.is_empty(_.difference(batch_indexes, mentions_by_page[2]))
      assert len(batch_indexes) == batch_size
    elif batch_num == 2:
      assert len(_.intersection(batch_indexes, mentions_by_page[2])) == 2
      assert len(_.intersection(batch_indexes, mentions_by_page[0])) == 1
      assert len(_.intersection(batch_indexes, mentions_by_page[1])) == 1
      assert len(batch_indexes) == 4
    batches_seen.append(batch_num)
  assert _.is_empty(_.difference(mentions_in_page_order, indexes_seen))
  assert batches_seen == [0, 1, 2]
