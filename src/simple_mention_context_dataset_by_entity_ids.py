from mention_context_dataset import MentionContextDataset
from data_transformers import get_mention_sentence_splits, embed_page_content
from parsers import parse_for_sentence_spans
import pydash as _

class SimpleMentionContextDatasetByEntityIds(MentionContextDataset):
  def __init__(self,
               cursor,
               page_id_order,
               entity_candidates_lookup,
               entity_label_lookup,
               embedding_lookup,
               batch_size,
               num_entities,
               num_mentions,
               num_candidates):
    super(SimpleMentionContextDatasetByEntityIds, self).__init__(cursor,
                                                                 page_id_order,
                                                                 entity_candidates_lookup,
                                                                 entity_label_lookup,
                                                                 embedding_lookup,
                                                                 batch_size,
                                                                 num_entities,
                                                                 num_mentions,
                                                                 num_candidates,
                                                                 transform=None)
    self.embedding_lookup = embedding_lookup
    self._page_content_lookup = {}
    self._sentence_spans_lookup = {}
    self._embedded_page_content_lookup = {}
    self._mention_infos = []
    entity_ids = self.entity_label_lookup.keys()
    for entity_id in entity_ids:
      self.cursor.execute('select mention, page_id, entity_id, mention_id, offset from entity_mentions_text where entity_id = %s', entity_id)
      mention_infos = self.cursor.fetchall()
      self._mention_infos.extend(mention_infos)
      page_ids = [mention['page_id'] for mention in mention_infos]
      for page_id in page_ids:
        self.cursor.execute('select * from pages where id = %s', page_id)
        page_content = self.cursor.fetchone()['content']
        page_mention_infos = [mention for mention in mention_infos if mention['page_id'] == page_id]
        self._page_content_lookup[page_id] = page_content
        self._embedded_page_content_lookup[page_id] = embed_page_content(self.embedding_lookup,
                                                                         page_mention_infos,
                                                                         page_content)
    self._sentence_spans_lookup = _.map_values(self._page_content_lookup, parse_for_sentence_spans)
    self._mention_infos = self._mention_infos[:num_mentions]

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
