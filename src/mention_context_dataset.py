import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pydash as _
import random

from data_transformers import get_mention_sentence_splits
from parsers import parse_for_sentences


class MentionContextDataset(Dataset):
  def __init__(self,
               cursor,
               page_id_order,
               entity_candidates_lookup,
               entity_label_lookup,
               batch_size,
               num_entities,
               num_mentions,
               transform=None):
    self.page_id_order = page_id_order
    self.entity_candidates_lookup = _.map_values(entity_candidates_lookup, lambda val: torch.tensor(val))
    self.entity_label_lookup = entity_label_lookup
    self.cursor = cursor
    self.transform = transform
    self.batch_size = batch_size
    self.num_mentions = num_mentions
    self.num_entities = num_entities
    self._page_id_sentences_lookup = {}
    self._document_mention_lookup = {}
    self._mention_infos = {}
    self._num_seen_mentions = 0
    self._num_candidates = 30
    self.page_ctr = 0

  def __len__(self):
    return self.num_mentions

  def __getitem__(self, idx):
    if self._num_seen_mentions == self.num_mentions: raise IndexError
    if idx not in self._mention_infos:
      print(idx, 'not in cache')
      self._next_batch()
    mention_info = self._mention_infos.pop(idx)
    sentences = self._page_id_sentences_lookup.pop(mention_info['page_id'])
    label = self.entity_label_lookup[mention_info['entity_id']]
    sample = {'sentence_splits': get_mention_sentence_splits(sentences, mention_info),
              'label': label,
              'document_mention_indices': self._document_mention_lookup.pop(mention_info['page_id']),
              'candidates': self._get_candidates(mention_info['mention'], label)}
    if self.transform:
      sample = self.transform(sample)
    self._num_seen_mentions += 1
    return sample

  def _get_candidates(self, mention, label):
    base_candidates = self.entity_candidates_lookup[mention]
    if len(base_candidates) < self._num_candidates:
      indexes_to_keep = range(len(base_candidates))
    else:
      label_index = int((base_candidates == label).nonzero().squeeze())
      indexes_to_sample = set(range(len(base_candidates))) - set([label_index])
      indexes_to_keep = random.sample(indexes_to_sample,
                                      self._num_candidates - 1) + [label_index]
    num_candidates_to_generate = self._num_candidates - len(indexes_to_keep)
    if num_candidates_to_generate != 0:
      random_candidates = torch.tensor(random.sample(range(self.num_entities), num_candidates_to_generate))
      return torch.cat((base_candidates[indexes_to_keep], random_candidates), 0)
    else:
      return base_candidates[indexes_to_keep]

  def _get_mention_infos_by_page_id(self, page_id):
    self.cursor.execute('select mention, page_id, entity_id, mention_id, offset from entity_mentions_text where page_id = %s', page_id)
    return self.cursor.fetchall()

  def _get_batch_mention_infos(self, closeby_page_ids):
    mention_infos = {}
    for page_id in closeby_page_ids:
      mentions = self._get_mention_infos_by_page_id(page_id)
      mention_infos.update({mention['mention_id']: mention for mention in mentions})
    return mention_infos

  def _get_batch_page_id_sentences_lookup(self, page_ids):
    lookup = {}
    for page_id in page_ids:
      self.cursor.execute('select content from pages where id = %s', page_id)
      lookup[page_id] = parse_for_sentences(self.cursor.fetchone()['content'])
    return lookup

  def _get_batch_document_mention_lookup(self, page_ids):
    lookup = {}
    for page_id in page_ids:
      self.cursor.execute('select id from mentions where page_id = %s', page_id)
      lookup[page_id] = [row['id'] for row in self.cursor.fetchall()]
    return lookup

  def _next_page_id_batch(self):
    num_mentions_in_batch = 0
    page_ids = []
    while num_mentions_in_batch < self.batch_size:
      self.cursor.execute('select count(*) from mentions where page_id = %s', self.page_ctr)
      num_mentions_in_batch += self.cursor.fetchone()['count(*)']
      page_ids.append(self.page_id_order[self.page_ctr])
    self.page_ctr += len(page_ids)
    return page_ids

  def _next_batch(self):
    print('getting batch')
    closeby_page_ids = self._next_page_id_batch()
    self._page_id_sentences_lookup.update(self._get_batch_page_id_sentences_lookup(closeby_page_ids))
    self._mention_infos.update(self._get_batch_mention_infos(closeby_page_ids))
    self._document_mention_lookup.update(self._get_batch_document_mention_lookup(closeby_page_ids))
