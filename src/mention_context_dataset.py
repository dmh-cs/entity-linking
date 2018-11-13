import Levenshtein

from torch.utils.data import Dataset
import torch

import pydash as _

from data_transformers import get_mention_sentence_splits, embed_page_content
from data_fetchers import get_candidate_ids
from parsers import parse_for_sentence_spans


class MentionContextDataset(Dataset):
  def __init__(self,
               cursor,
               page_id_order,
               entity_candidates_prior,
               entity_label_lookup,
               embedding,
               token_idx_lookup,
               batch_size,
               num_entities,
               num_candidates,
               cheat=False):
    self.page_id_order = page_id_order
    self.entity_candidates_prior = entity_candidates_prior
    self.entity_label_lookup = _.map_values(entity_label_lookup, torch.tensor)
    self.embedding = embedding
    self.token_idx_lookup = token_idx_lookup
    self.cursor = cursor
    self.batch_size = batch_size
    self.num_entities = num_entities
    self.num_candidates = num_candidates
    self._sentence_spans_lookup = {}
    self._page_content_lookup = {}
    self._embedded_page_content_lookup = {}
    self._entity_page_mentions_lookup = {}
    self._mentions_per_page_ctr = {}
    self._mention_infos = {}
    self.page_ctr = 0
    self.cheat = cheat

  def _get_candidate_ids(self, mention, label):
    return get_candidate_ids(self.entity_candidates_prior,
                             self.num_entities,
                             self.num_candidates,
                             mention,
                             label,
                             cheat=self.cheat)

  def _get_candidate_strs(self, candidate_ids):
    self.cursor.execute('select text from entities where page_id in (%s)', ', '.join(candidate_ids))
    return [row['text'] for row in self.cursor.fetchall()]

  def __len__(self):
    raise NotImplementedError

  def __getitem__(self, idx):
    if idx not in self._mention_infos:
      self._next_batch()
    mention_info = self._mention_infos.pop(idx)
    sentence_spans = self._sentence_spans_lookup[mention_info['page_id']]
    page_content = self._page_content_lookup[mention_info['page_id']]
    label = self.entity_label_lookup[mention_info['entity_id']]
    candidate_ids = self._get_candidate_ids(mention_info['mention'], label)
    p_prior = self._get_p_prior(mention_info['mention'], candidate_ids)
    candidates = self._get_candidate_strs(candidate_ids)
    sample = {'sentence_splits': get_mention_sentence_splits(page_content,
                                                             sentence_spans,
                                                             mention_info),
              'label': label,
              'embedded_page_content': self._embedded_page_content_lookup[mention_info['page_id']],
              'entity_page_mentions': self._entity_page_mentions_lookup[mention_info['page_id']],
              'p_prior': p_prior,
              'candidate_ids': candidate_ids,
              'candidate_mention_sim': torch.tensor([Levenshtein.ratio(mention_info['mention'], candidate)
                                                     for candidate in candidates])}
    self._mentions_per_page_ctr[mention_info['page_id']] -= 1
    if self._mentions_per_page_ctr[mention_info['page_id']] == 0:
      self._sentence_spans_lookup.pop(mention_info['page_id'])
      self._page_content_lookup.pop(mention_info['page_id'])
      self._embedded_page_content_lookup.pop(mention_info['page_id'])
      self._entity_page_mentions_lookup.pop(mention_info['page_id'])
    return sample

  def _get_p_prior(self, mention, candidate_ids):
    if mention not in self.entity_candidates_prior:
      return torch.zeros(len(candidate_ids))
    entity_counts = self.entity_candidates_prior[mention]
    candidate_counts = [entity_counts[entity] if entity in entity_counts else 0 for entity in candidate_ids.tolist()]
    return torch.tensor(candidate_counts, dtype=torch.float) / sum(candidate_counts)

  def _get_mention_infos_by_page_id(self, page_id):
    self.cursor.execute('select mention, page_id, entity_id, mention_id, offset from entity_mentions_text where page_id = %s', page_id)
    return self.cursor.fetchall()

  def _get_batch_mention_infos(self, closeby_page_ids):
    mention_infos = {}
    for page_id in closeby_page_ids:
      mentions = self._get_mention_infos_by_page_id(page_id)
      self._mentions_per_page_ctr[page_id] = len(mentions)
      mention_infos.update({mention['mention_id']: mention for mention in mentions})
    return mention_infos

  def _get_batch_sentence_spans_lookup(self, page_ids):
    lookup = {}
    for page_id in page_ids:
      self.cursor.execute('select content from pages where id = %s', page_id)
      lookup[page_id] = parse_for_sentence_spans(self.cursor.fetchone()['content'])
    return lookup

  def _get_batch_page_content_lookup(self, page_ids):
    lookup = {}
    for page_id in page_ids:
      self.cursor.execute('select content from pages where id = %s', page_id)
      lookup[page_id] = self.cursor.fetchone()['content']
    return lookup

  def _get_batch_entity_page_mentions_lookup(self, page_ids):
    lookup = {}
    for page_id in page_ids:
      page_mention_infos = list(filter(lambda mention_info: mention_info['page_id'] == page_id,
                                       self._mention_infos.values()))
      content = ' '.join([mention_info['mention'] for mention_info in page_mention_infos])
      if _.is_empty(page_mention_infos):
        lookup[page_id] = torch.tensor([])
      else:
        lookup[page_id] = embed_page_content(self.embedding,
                                             self.token_idx_lookup,
                                             content,
                                             page_mention_infos)
    return lookup

  def _get_batch_embedded_page_content_lookup(self, page_ids):
    lookup = {}
    for page_id in page_ids:
      page_content = self._page_content_lookup[page_id]
      if len(page_content.strip()) > 5:
        lookup[page_id] = embed_page_content(self.embedding,
                                             self.token_idx_lookup,
                                             page_content)
    return lookup

  def _next_page_id_batch(self):
    num_mentions_in_batch = 0
    page_ids = []
    while num_mentions_in_batch < self.batch_size and self.page_ctr < len(self.page_id_order):
      page_id_to_add = self.page_id_order[self.page_ctr]
      self.cursor.execute('select count(*) from mentions where page_id = %s', page_id_to_add)
      num_mentions_in_batch += self.cursor.fetchone()['count(*)']
      page_ids.append(page_id_to_add)
      self.page_ctr += 1
    return page_ids

  def _next_batch(self):
    closeby_page_ids = self._next_page_id_batch()
    self._sentence_spans_lookup.update(self._get_batch_sentence_spans_lookup(closeby_page_ids))
    self._page_content_lookup.update(self._get_batch_page_content_lookup(closeby_page_ids))
    self._mention_infos.update(self._get_batch_mention_infos(closeby_page_ids))
    self._entity_page_mentions_lookup.update(self._get_batch_entity_page_mentions_lookup(closeby_page_ids))
    self._embedded_page_content_lookup.update(self._get_batch_embedded_page_content_lookup(closeby_page_ids))
