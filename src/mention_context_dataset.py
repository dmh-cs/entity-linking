import Levenshtein
from collections import defaultdict

from torch.utils.data import Dataset
import torch

import pydash as _

from data_transformers import get_mention_sentence_splits, embed_page_content
from data_fetchers import get_candidate_ids, get_p_prior, get_candidate_strs
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
               cheat=False,
               buffer_scale=1):
    self.page_id_order = page_id_order
    self.entity_candidates_prior = entity_candidates_prior
    self.entity_label_lookup = _.map_values(entity_label_lookup, torch.tensor)
    self.entity_id_lookup = {int(label): entity_id for entity_id, label in self.entity_label_lookup.items()}
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
    self._candidate_strs_lookup = {}
    self.page_ctr = 0
    self.cheat = cheat
    self.buffer_scale = buffer_scale

  def _get_candidate_ids(self, mention, label):
    return get_candidate_ids(self.entity_candidates_prior,
                             self.num_entities,
                             self.num_candidates,
                             mention,
                             label,
                             cheat=self.cheat)

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
    p_prior = get_p_prior(self.entity_candidates_prior, mention_info['mention'], candidate_ids)
    candidates = self._get_candidate_strs(candidate_ids.tolist())
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

  def _get_candidate_strs(self, candidate_ids):
    return [self._candidate_strs_lookup[candidate_id]
            if candidate_id in self._candidate_strs_lookup else ''
            for candidate_id in candidate_ids]

  def _get_mention_infos_by_page_id(self, page_ids):
    self.cursor.execute('select mention, page_id, entity_id, mention_id, offset from entity_mentions_text where page_id in (' + str(page_ids)[1:-1] + ')')
    rows = self.cursor.fetchall()
    result = defaultdict(list)
    for row in rows:
      result[row['page_id']].append(row)
    return dict(result)

  def _get_batch_mention_infos(self, closeby_page_ids):
    self._candidate_strs_lookup = {}
    mention_infos = {}
    mentions_covered = set()
    mentions_by_page_id = self._get_mention_infos_by_page_id(closeby_page_ids)
    candidate_ids = []
    for page_id, mentions in mentions_by_page_id.items():
      for mention_info in mentions:
        if mention_info['mention'] in mentions_covered: continue
        mentions_covered.add(mention_info['mention'])
        label = self.entity_label_lookup[mention_info['entity_id']]
        candidate_ids.extend(self._get_candidate_ids(mention_info['mention'], label).tolist())
      self._mentions_per_page_ctr[page_id] = len(mentions)
      mention_infos.update({mention['mention_id']: mention for mention in mentions})
    self._candidate_strs_lookup.update(dict(zip(candidate_ids,
                                                get_candidate_strs(self.cursor,
                                                                   [self.entity_id_lookup[cand_id] for cand_id in candidate_ids]))))
    return mention_infos

  def _to_sentence_spans_lookup(self, content_lookup):
    lookup = {}
    for page_id, content in content_lookup.items():
      lookup[page_id] = parse_for_sentence_spans(content)
    return lookup

  def _get_batch_page_content_lookup(self, page_ids):
    lookup = {}
    self.cursor.execute('select id, content from pages where id in (' + str(page_ids)[1:-1] + ')')
    for row in self.cursor.fetchall():
      lookup[row['id']] = row['content']
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
    while num_mentions_in_batch < self.batch_size * self.buffer_scale and self.page_ctr < len(self.page_id_order):
      page_id_to_add = self.page_id_order[self.page_ctr]
      self.cursor.execute('select count(*) from mentions where page_id = %s', page_id_to_add)
      num_mentions_in_batch += self.cursor.fetchone()['count(*)']
      page_ids.append(page_id_to_add)
      self.page_ctr += 1
    return page_ids

  def _next_batch(self):
    closeby_page_ids = self._next_page_id_batch()
    page_content = self._get_batch_page_content_lookup(closeby_page_ids)
    self._page_content_lookup.update(page_content)
    self._sentence_spans_lookup.update(self._to_sentence_spans_lookup(page_content))
    self._mention_infos.update(self._get_batch_mention_infos(closeby_page_ids))
    self._entity_page_mentions_lookup.update(self._get_batch_entity_page_mentions_lookup(closeby_page_ids))
    self._embedded_page_content_lookup.update(self._get_batch_embedded_page_content_lookup(closeby_page_ids))
