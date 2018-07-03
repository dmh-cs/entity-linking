from mention_context_dataset import MentionContextDataset
from data_transformers import get_mention_sentence_splits
from parsers import parse_for_sentence_spans
import pydash as _

class SimpleMentionContextDataset(MentionContextDataset):
  def __init__(self,
               cursor,
               page_id_order,
               entity_candidates_lookup,
               entity_label_lookup,
               batch_size,
               num_entities,
               num_mentions,
               num_candidates):
    super(SimpleMentionContextDataset, self).__init__(cursor,
                                                      page_id_order,
                                                      entity_candidates_lookup,
                                                      entity_label_lookup,
                                                      batch_size,
                                                      num_entities,
                                                      num_mentions,
                                                      num_candidates,
                                                      transform=None)
    self._page_content_lookup = {}
    self._sentence_spans_lookup = {}
    self._document_mention_lookup = {}
    self._mention_infos = []
    page_ids = iter(page_id_order)
    while len(self._mention_infos) < num_mentions:
      page_id = next(page_ids)
      self.cursor.execute('select * from pages where id = %s', page_id)
      self._page_content_lookup.update({page_id: self.cursor.fetchone()['content']})
      self._sentence_spans_lookup.update(_.map_values(self._page_content_lookup, parse_for_sentence_spans))
      self.cursor.execute('select id from mentions where page_id = %s', page_id)
      self._document_mention_lookup.update({page_id: [row['id'] for row in self.cursor.fetchall()]})
      self.cursor.execute('select mention, page_id, entity_id, mention_id, offset from entity_mentions_text where page_id = %s', page_id)
      self._mention_infos.extend(self.cursor.fetchall())

  def __len__(self):
    return self.num_mentions

  def __getitem__(self, idx):
    mention_info = self._mention_infos[idx]
    sentence_spans = self._sentence_spans_lookup[mention_info['page_id']]
    page_content = self._page_content_lookup[mention_info['page_id']]
    label = self.entity_label_lookup[mention_info['entity_id']]
    sample = {'sentence_splits': get_mention_sentence_splits(page_content,
                                                             sentence_spans,
                                                             mention_info),
              'label': label,
              'document_mention_indices': self._document_mention_lookup[mention_info['page_id']],
              'candidates': self._get_candidates(mention_info['mention'], label)}
    return sample
