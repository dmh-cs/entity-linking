import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pydash as _

from data_transformers import get_mention_sentence_splits
from parsers import parse_for_sentences


class MentionContextDataset(Dataset):
  def __init__(self,
               cursor,
               page_id_order,
               entity_candidates_lookup,
               batch_size,
               num_mentions,
               transform=None):
    self.page_id_order = page_id_order
    self.entity_candidates_lookup = entity_candidates_lookup
    self.cursor = cursor
    self.transform = transform
    self.batch_size = batch_size
    self.num_mentions = num_mentions
    self._page_id_sentences_lookup = {}
    self._document_mention_lookup = {}
    self._mention_infos = {}
    self._entity_label_ctr = 0
    self.entity_label_lookup = {}
    self.page_ctr = 0

  def __len__(self):
    return self.num_mentions - 1

  def __getitem__(self, idx):
    if idx > len(self): raise IndexError
    if idx not in self._mention_infos:
      self._next_batch(self.batch_size)
    mention_info = self._mention_infos[idx]
    sentences = self._page_id_sentences_lookup[mention_info['page_id']]
    if mention_info['entity_id'] not in self.entity_label_lookup:
      self.entity_label_lookup[mention_info['entity_id']] = self._entity_label_ctr
      self._entity_label_ctr += 1
    sample = {'sentence_splits': get_mention_sentence_splits(sentences, mention_info),
              'label': self.entity_label_lookup[mention_info['entity_id']],
              'document_mention_indices': self._document_mention_lookup[mention_info['entity_id']],
              'candidates': self.entity_candidates_lookup[mention_info['mention_id']]}
    if self.transform:
      sample = self.transform(sample)
    return sample

  def _get_mention_infos_by_page_id(self, page_id):
    self.cursor.execute('select mention, page_id, entity_id, mention_id, offset from entity_mentions_text where page_id = %s', page_id)
    return self.cursor.fetchall()

  def _get_batch_mention_infos(self, closeby_page_ids, batch_size):
    mention_infos = []
    while len(mention_infos) < batch_size:
      mention_infos.extend(self._get_mention_infos_by_page_id(closeby_page_ids[self.page_ctr]))
      self.page_ctr += 1
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

  def _next_batch(self, batch_size):
    closeby_page_ids = self.page_id_order[self.page_ctr : self.page_ctr + batch_size]
    self._page_id_sentences_lookup = self._get_batch_page_id_sentences_lookup(closeby_page_ids)
    self._mention_infos = self._get_batch_mention_infos(closeby_page_ids, batch_size)
    self._document_mention_lookup = self._get_batch_document_mention_lookup(closeby_page_ids)
