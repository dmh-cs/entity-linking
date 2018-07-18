from torch.utils.data import Dataset, DataLoader
from mention_context_dataset import MentionContextDataset
from data_transformers import get_mention_sentence_splits, embed_page_content
from data_fetchers import get_candidates
from parsers import parse_for_sentence_spans
import torch
import pydash as _

class SimpleMentionContextDatasetByEntityIds(Dataset):
  def __init__(self,
               cursor,
               entity_candidates_lookup,
               entity_label_lookup,
               embedding_lookup,
               num_candidates,
               entity_ids,
               train=True):
    self.cursor = cursor
    self.entity_candidates_lookup = _.map_values(entity_candidates_lookup, torch.tensor)
    self.entity_label_lookup = _.map_values(entity_label_lookup, torch.tensor)
    self.embedding_lookup = embedding_lookup
    self.num_candidates = num_candidates
    self.num_entities = len(entity_ids)
    self._page_content_lookup = {}
    self._sentence_spans_lookup = {}
    self._embedded_page_content_lookup = {}
    self._mention_infos = []
    print('Getting', len(entity_ids), 'entities and their info')
    for entity_id in entity_ids:
      self.cursor.execute('select mention, page_id, entity_id, mention_id, offset from entity_mentions_text where entity_id = %s', entity_id)
      mention_infos = _.sort_by(self.cursor.fetchall(), 'mention_id')
      if train:
        mention_infos = mention_infos[:int(0.8 * len(mention_infos))]
      else:
        mention_infos = mention_infos[int(0.8 * len(mention_infos)):]
      self._mention_infos.extend(mention_infos)
      page_ids = [mention['page_id'] for mention in mention_infos]
      for page_id in page_ids:
        if page_id in self._embedded_page_content_lookup:
          continue
        self.cursor.execute('select * from pages where id = %s', page_id)
        page_content = self.cursor.fetchone()['content']
        page_mention_infos = [mention for mention in mention_infos if mention['page_id'] == page_id]
        self._page_content_lookup[page_id] = page_content
        if not _.is_empty(page_content):
          self._embedded_page_content_lookup[page_id] = embed_page_content(self.embedding_lookup,
                                                                           page_mention_infos,
                                                                           page_content)
    self._sentence_spans_lookup = _.map_values(self._page_content_lookup, parse_for_sentence_spans)

  def _get_candidates(self, mention, label):
    return get_candidates(self.entity_candidates_lookup,
                          self.num_entities,
                          self.num_candidates,
                          mention,
                          label)

  def _get_batch_embedded_page_content_lookup(self, page_ids):
    lookup = {}
    for page_id in page_ids:
      page_mention_infos = filter(lambda mention_info: mention_info['page_id'] == page_id,
                                  self._mention_infos)
      page_content = self._page_content_lookup[page_id]
      if len(page_content.strip()) > 5:
        lookup[page_id] = embed_page_content(self.embedding_lookup,
                                             page_mention_infos,
                                             page_content)
    return lookup

  def __len__(self):
    return len(self._mention_infos)

  def __getitem__(self, idx):
    mention_info = self._mention_infos[idx]
    sentence_spans = self._sentence_spans_lookup[mention_info['page_id']]
    page_content = self._page_content_lookup[mention_info['page_id']]
    label = self.entity_label_lookup[mention_info['entity_id']]
    sample = {'sentence_splits': get_mention_sentence_splits(page_content,
                                                             sentence_spans,
                                                             mention_info),
              'label': label,
              'embedded_page_content': self._embedded_page_content_lookup[mention_info['page_id']],
              'candidates': self._get_candidates(mention_info['mention'], label)}
    return sample
