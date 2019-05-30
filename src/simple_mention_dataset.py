import pydash as _
import Levenshtein
from collections import Counter
import torch
import pickle
from nltk.stem.snowball import SnowballStemmer

from conll_helpers import get_documents, get_mentions, get_splits, get_entity_page_ids, from_page_ids_to_entity_ids, get_doc_id_per_mention, get_mentions_by_doc_id, get_mention_sentences
from data_fetchers import load_entity_candidate_ids_and_label_lookup, get_candidate_ids_simple, get_p_prior, get_candidate_strs
from parsers import parse_text_for_tokens
from data_transformers import get_mention_sentences_from_infos
import utils as u

class SimpleMentionDataset():
  def __init__(self, cursor, page_ids, lookups_path, train_size):
    self.stemmer = SnowballStemmer('english')
    self.page_content_lim = 2000
    self.cursor = cursor
    self.page_ids = page_ids
    self.document_lookup = self.get_document_lookup(page_ids)
    self.mention_infos = self.get_mention_infos(page_ids)
    self.mentions = [info['mention'] for info in self.mention_infos]
    self.labels = [info['entity_id'] for info in self.mention_infos]
    self.mention_doc_id = [info['page_id'] for info in self.mention_infos]
    self.with_label = [i for i, x in enumerate(self.labels) if x != -1]
    self.mention_sentences = get_mention_sentences_from_infos(self.document_lookup, self.mention_infos)
    self.mention_fs = [dict(Counter(self.stemmer.stem(token)
                                    for token in parse_text_for_tokens(sentence)))
                       for sentence in self.mention_sentences]
    self.page_token_cnts_lookup = {page_id: dict(Counter(self.stemmer.stem(token)
                                                         for token in parse_text_for_tokens(doc[:self.page_content_lim])))
                                   for page_id, doc in self.document_lookup.items()}
    lookups = load_entity_candidate_ids_and_label_lookup(lookups_path, train_size)
    label_to_entity_id = _.invert(lookups['entity_labels'])
    self.entity_candidates_prior = {entity_text: {label_to_entity_id[label]: candidates
                                                  for label, candidates in prior.items()}
                                    for entity_text, prior in lookups['entity_candidates_prior'].items()}
    self.prior_approx_mapping = u.get_prior_approx_mapping(self.entity_candidates_prior)

  def get_document_lookup(self, page_ids):
    self.cursor.execute(f'select id, left(content, {self.page_content_lim}) as content from pages where id in (' + str(page_ids)[1:-1] + ')')
    return {row['id']: row['content'] for row in self.cursor.fetchall()}

  def get_mention_infos(self, page_ids):
    self.cursor.execute('select mention, page_id, entity_id, mention_id, offset from entity_mentions_text where page_id in (' + str(page_ids)[1:-1] + ')')
    return self.cursor.fetchall()

  def __getitem__(self, idx):
    label = self.labels[idx]
    mention = self.mentions[idx]
    mention_f = self.mention_fs[idx]
    page_token_cnts = self.page_token_cnts_lookup[self.mention_doc_id[idx]]
    candidate_ids = get_candidate_ids_simple(self.entity_candidates_prior,
                                             self.prior_approx_mapping,
                                             mention)
    candidates = get_candidate_strs(self.cursor, candidate_ids.tolist())
    p_prior = get_p_prior(self.entity_candidates_prior,
                          self.prior_approx_mapping,
                          mention,
                          candidate_ids)
    candidate_mention_sim = torch.tensor([Levenshtein.ratio(mention, candidate)
                                          for candidate in candidates])
    return {'mention_f': mention_f,
            'label': label,
            'page_token_cnts': page_token_cnts,
            'p_prior': p_prior,
            'candidate_ids': candidate_ids.tolist(),
            'candidate_strs': candidates,
            'candidate_mention_sim': candidate_mention_sim}
