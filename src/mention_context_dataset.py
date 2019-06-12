import Levenshtein
from collections import defaultdict, Counter
from nltk.stem.snowball import SnowballStemmer
import unidecode

from torch.utils.data import Dataset
import torch

import pydash as _

from data_transformers import get_mention_sentence_splits, embed_page_content, get_bag_of_nouns, tokens_to_embeddings, get_mention_sentence
from data_fetchers import get_candidate_ids, get_p_prior, get_candidate_strs
from parsers import parse_for_sentence_spans, parse_text_for_tokens
import utils as u
from cache import read_cache


def _get_str_lookup(cursor, entity_label_lookup):
  cursor.execute('select id, text from entities')
  return {entity_label_lookup[row['id']]: row['text']
          for row in cursor.fetchall() if row['id'] in entity_label_lookup}

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
               entity_embeds,
               cheat=False,
               buffer_scale=1,
               min_mentions=1,
               use_fast_sampler=False,
               use_wiki2vec=False,
               use_sum_encoder=False,
               start_from_page_num=0,
               ablation=['local_context', 'document_context', 'prior']):
    self._candidate_strs_lookup = read_cache('./candidate_strs_lookup.pkl',
                                             lambda: _get_str_lookup(cursor, entity_label_lookup))
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
    self._page_token_cnts_lookup = {}
    self._entity_page_mentions_lookup = {}
    self._mentions_per_page_ctr = defaultdict(int)
    self._mention_infos = {}
    self._bag_of_nouns_lookup = {}
    self.page_ctr = start_from_page_num
    self.cheat = cheat
    self.buffer_scale = buffer_scale
    self.min_mentions = min_mentions
    self.use_fast_sampler = use_fast_sampler
    self.use_wiki2vec = use_wiki2vec
    self.use_sum_encoder = use_sum_encoder
    # if self.use_fast_sampler: assert not self.use_wiki2vec, 'train wiki2vec locally'
    self.prior_approx_mapping = u.get_prior_approx_mapping(self.entity_candidates_prior)
    self.page_content_lim = 5000
    if self.min_mentions > 1:
      query = 'select id from entities where num_mentions >= ' + str(self.min_mentions)
      cursor.execute(query)
      self.valid_entity_ids = set(row['id'] for row in cursor.fetchall())
    self.ablation = ablation
    self.entity_embeds = entity_embeds
    self.stemmer = SnowballStemmer('english')
    self._offset = 0

  def _get_candidate_ids(self, mention, label):
    return get_candidate_ids(self.entity_candidates_prior,
                             self.prior_approx_mapping,
                             self.num_entities,
                             self.num_candidates,
                             mention,
                             label,
                             cheat=self.cheat)

  def __len__(self):
    raise NotImplementedError

  def _getitem(self, idx):
    if self.use_fast_sampler:
      if len(self._mention_infos) == 0: self._next_batch()
      idx = next(iter(self._mention_infos.keys()))
    if idx not in self._mention_infos:
      self._next_batch()
    mention_info = self._mention_infos.pop(idx)
    sentence_spans = self._sentence_spans_lookup[mention_info['page_id']]
    page_content = self._page_content_lookup[mention_info['page_id']]
    label = self.entity_label_lookup[mention_info['entity_id']]
    candidate_ids = self._get_candidate_ids(mention_info['mention'], label)
    p_prior = get_p_prior(self.entity_candidates_prior, self.prior_approx_mapping, mention_info['mention'], candidate_ids)
    candidates = self._get_candidate_strs(candidate_ids.tolist())
    sample = {'sentence_splits': get_mention_sentence_splits(page_content,
                                                             sentence_spans,
                                                             mention_info,
                                                             lim=self.page_content_lim),
              'label': label,
              'embedded_page_content': self._embedded_page_content_lookup[mention_info['page_id']],
              'entity_page_mentions': self.entity_embeds(self._entity_page_mentions_lookup[mention_info['page_id']]),
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

  def _sum_encoder_getitem(self, idx):
    if self.use_fast_sampler:
      if len(self._mention_infos) == 0: self._next_batch()
      idx = next(iter(self._mention_infos.keys()))
    if idx not in self._mention_infos:
      self._next_batch()
    mention_info = self._mention_infos.pop(idx)
    sentence_spans = self._sentence_spans_lookup[mention_info['page_id']]
    page_content = self._page_content_lookup[mention_info['page_id']]
    label = self.entity_label_lookup[mention_info['entity_id']]
    candidate_ids = self._get_candidate_ids(mention_info['mention'], label)
    p_prior = get_p_prior(self.entity_candidates_prior, self.prior_approx_mapping, mention_info['mention'], candidate_ids)
    candidates = self._get_candidate_strs(candidate_ids.tolist())
    sample = {'mention_sentence': get_mention_sentence(page_content,
                                                       sentence_spans,
                                                       mention_info,
                                                       lim=self.page_content_lim),
              'label': label,
              'page_token_cnts': self._page_token_cnts_lookup[mention_info['page_id']],
              'p_prior': p_prior,
              'candidate_ids': candidate_ids,
              'candidate_mention_sim': torch.tensor([Levenshtein.ratio(mention_info['mention'], candidate)
                                                     for candidate in candidates])}
    self._mentions_per_page_ctr[mention_info['page_id']] -= 1
    if self._mentions_per_page_ctr[mention_info['page_id']] == 0:
      self._sentence_spans_lookup.pop(mention_info['page_id'])
      self._page_content_lookup.pop(mention_info['page_id'])
      self._page_token_cnts_lookup.pop(mention_info['page_id'])
    return sample

  def _wiki2vec_getitem(self, idx):
    if self.use_fast_sampler:
      if len(self._mention_infos) == 0: self._next_batch()
      idx = next(iter(self._mention_infos.keys()))
    if idx not in self._mention_infos:
      self._next_batch()
    mention_info = self._mention_infos.pop(idx)
    bag_of_nouns = self._bag_of_nouns_lookup[mention_info['page_id']]
    label = self.entity_label_lookup[mention_info['entity_id']]
    candidate_ids = self._get_candidate_ids(mention_info['mention'], label)
    p_prior = get_p_prior(self.entity_candidates_prior, self.prior_approx_mapping, mention_info['mention'], candidate_ids)
    candidates = self._get_candidate_strs(candidate_ids.tolist())
    sample = {'bag_of_nouns': bag_of_nouns,
              'label': label,
              'p_prior': p_prior,
              'candidate_ids': candidate_ids,
              'candidate_mention_sim': torch.tensor([Levenshtein.ratio(mention_info['mention'], candidate)
                                                     for candidate in candidates])}
    self._mentions_per_page_ctr[mention_info['page_id']] -= 1
    if self._mentions_per_page_ctr[mention_info['page_id']] == 0:
      self._bag_of_nouns_lookup.pop(mention_info['page_id'])
    return sample

  def __getitem__(self, idx):
    if self.use_wiki2vec:
      return self._wiki2vec_getitem(idx)
    elif self.use_sum_encoder:
      return self._sum_encoder_getitem(idx)
    else:
      return self._getitem(idx)

  def _get_candidate_strs(self, candidate_ids):
    return [self._candidate_strs_lookup[candidate_id]
            if candidate_id in self._candidate_strs_lookup else ''
            for candidate_id in candidate_ids]

  def _get_mention_infos_by_page_id(self, page_ids):
    self.cursor.execute('select mention, page_id, entity_id, mention_id, offset from entity_mentions_text where page_id in (' + str(page_ids)[1:-1] + ')')
    rows = self.cursor.fetchall()
    result = defaultdict(list)
    for row in rows:
      if (self.min_mentions == 1) or (row['entity_id'] in self.valid_entity_ids):
        result[row['page_id']].append(row)
    return dict(result)

  def _get_batch_mention_infos(self):
    num_to_fetch = self.batch_size
    mention_infos = {}
    while len(mention_infos) < self.batch_size:
      self.cursor.execute(f'select mention, page_id, entity_id, mention_id, offset from entity_mentions_text limit {num_to_fetch} offset {self._offset}')
      rows = self.cursor.fetchall()
      for row in rows:
        if (self.min_mentions == 1) or (row['entity_id'] in self.valid_entity_ids):
          self._mentions_per_page_ctr[row['page_id']] += 1
          mention_infos[row['mention_id']] = row
          num_to_fetch -= -1
        self._offset += 1
    return mention_infos

  def _to_sentence_spans_lookup(self, content_lookup):
    lookup = {}
    for page_id, content in content_lookup.items():
      lookup[page_id] = parse_for_sentence_spans(content)
    return lookup

  def _get_batch_bag_of_nouns_lookup(self, content_lookup):
    lookup = {}
    for page_id, content in content_lookup.items():
      lookup[page_id] = get_bag_of_nouns(content)
    return lookup

  def _get_batch_page_content_lookup(self, page_ids):
    lookup = {}
    self.cursor.execute('select id, content from pages where id in (' + str(page_ids)[1:-1] + ')')
    for row in self.cursor.fetchall():
      lookup[row['id']] = row['content']
    return lookup

  def _get_batch_entity_page_mentions_lookup(self, page_ids):
    lookup = {}
    page_mention_infos_lookup = defaultdict(list)
    for mention_info in self._mention_infos.values():
      page_mention_infos_lookup[mention_info['page_id']].append(mention_info)
    for page_id in page_ids:
      page_mention_infos = page_mention_infos_lookup[page_id]
      entity_ids_for_page = []
      for mention_info in page_mention_infos:
        mention = mention_info['mention']
        if self.entity_candidates_prior.get(mention) is None:
          approx_mentions = self.prior_approx_mapping.get(unidecode.unidecode(mention).lower(), [])
          candidate_ids = list(set(sum([list(self.entity_candidates_prior.get(approx_mention, {}).keys())
                              for approx_mention in approx_mentions], [])))
        else:
          candidate_ids = list(self.entity_candidates_prior[mention].keys())
        prior = get_p_prior(self.entity_candidates_prior, self.prior_approx_mapping, mention_info['mention'], torch.tensor(candidate_ids))
        if len(prior) > 0:
          most_common_idx = int(torch.argmax(prior))
          entity_ids_for_page.append(candidate_ids[most_common_idx])
      lookup[page_id] = torch.tensor(entity_ids_for_page,
                                     device=self.entity_embeds.weight.device)
    return lookup

  def _get_batch_embedded_page_content_lookup(self, page_ids):
    lim = self.page_content_lim
    lookup = {}
    for page_id in page_ids:
      page_content = self._page_content_lookup[page_id]
      if len(page_content.strip()) > 5:
        lookup[page_id] = embed_page_content(self.embedding,
                                             self.token_idx_lookup,
                                             page_content[:lim])
    return lookup

  def _get_batch_page_token_cnts_lookup(self, page_ids):
    lim = self.page_content_lim
    lookup = {}
    for page_id in page_ids:
      page_content = self._page_content_lookup[page_id]
      if len(page_content.strip()) > 5:
        lookup[page_id] = dict(Counter(u.to_idx(self.token_idx_lookup, self._stem(token))
                                       for token in parse_text_for_tokens(page_content[:lim])))
    return lookup

  def _next_page_id_batch(self):
    num_mentions_in_batch = 0
    page_ids = []
    num_page_batch_size = 10000
    while num_mentions_in_batch < self.batch_size * self.buffer_scale and self.page_ctr < len(self.page_id_order):
      if self.min_mentions > 1:
        page_ids_to_add = self.page_id_order[self.page_ctr : self.page_ctr + num_page_batch_size]
        self.cursor.execute(f'select count(*) from (select 1 from mentions m inner join entity_mentions em on m.id = em.mention_id inner join entities e on e.id = em.entity_id where m.page_id in ({str(page_ids_to_add)[1:-1]}) and e.num_mentions > {self.min_mentions}) tab')
        num_mentions_in_batch += self.cursor.fetchone()['count(*)']
        page_ids.extend(page_ids_to_add)
        self.page_ctr += num_page_batch_size
      else:
        page_id_to_add = self.page_id_order[self.page_ctr]
        self.cursor.execute('select count(*) from mentions where page_id = %s', page_id_to_add)
        num_mentions_in_batch += self.cursor.fetchone()['count(*)']
        page_ids.append(page_id_to_add)
        self.page_ctr += 1
    return page_ids

  def _stem(self, text):
    return self.stemmer.stem(text)

  def _next_batch(self):
    new_mention_infos = self._get_batch_mention_infos()
    self._mention_infos.update(new_mention_infos)
    closeby_page_ids = [mention_info['page_id'] for mention_info in new_mention_infos.values()]
    page_content = self._get_batch_page_content_lookup(closeby_page_ids)
    if self.use_sum_encoder:
      self._page_content_lookup.update(page_content)
      self._sentence_spans_lookup.update(self._to_sentence_spans_lookup(page_content))
      self._page_token_cnts_lookup.update(self._get_batch_page_token_cnts_lookup(closeby_page_ids))
    elif not self.use_wiki2vec:
      self._page_content_lookup.update(page_content)
      self._sentence_spans_lookup.update(self._to_sentence_spans_lookup(page_content))
      self._entity_page_mentions_lookup.update(self._get_batch_entity_page_mentions_lookup(closeby_page_ids))
      self._embedded_page_content_lookup.update(self._get_batch_embedded_page_content_lookup(closeby_page_ids))
    else:
      self._bag_of_nouns_lookup.update(self._get_batch_bag_of_nouns_lookup(page_content))
