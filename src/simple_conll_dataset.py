import pydash as _
import Levenshtein
from collections import Counter
import torch

from conll_helpers import get_documents, get_mentions, get_splits, get_entity_page_ids, from_page_ids_to_entity_ids, get_doc_id_per_mention, get_mentions_by_doc_id, get_mention_sentences
from data_fetchers import load_entity_candidate_ids_and_label_lookup, get_candidate_ids_simple, get_p_prior, get_candidate_strs
from parsers import parse_text_for_tokens
import utils as u

class SimpleCoNLLDataset():
  def __init__(self, cursor, conll_path, lookups_path, train_size):
    self.cursor = cursor
    with open(conll_path, 'r') as fh:
      lines = fh.read().strip().split('\n')[:-1]
    self.documents = get_documents(lines)
    self.mentions = get_mentions(lines)
    self.sentence_splits = get_splits(self.documents, self.mentions)
    self.entity_page_ids = get_entity_page_ids(lines)
    self.labels = from_page_ids_to_entity_ids(self.cursor, self.entity_page_ids)
    self.with_label = [i for i, x in enumerate(self.labels) if x != -1]
    self.mention_doc_id = get_doc_id_per_mention(lines)
    self.mentions_by_doc_id = get_mentions_by_doc_id(lines)
    self.mention_sentences = get_mention_sentences(self.documents, self.mentions)
    self.page_token_cnts_lookup = [dict(Counter(parse_text_for_tokens(doc)))
                                   for doc in self.documents]
    lookups = load_entity_candidate_ids_and_label_lookup(lookups_path, train_size)
    label_to_entity_id = _.invert(lookups['entity_labels'])
    self.entity_candidates_prior = {entity_text: {label_to_entity_id[label]: candidates
                                                  for label, candidates in prior.items()}
                                    for entity_text, prior in lookups['entity_candidates_prior'].items()}
    self.prior_approx_mapping = u.get_prior_approx_mapping(self.entity_candidates_prior)

  def __iter__(self):
    for idx in self.with_label:
      label = self.labels[idx]
      mention = self.mentions[idx]
      mention_sentence = self.mention_sentences[idx]
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
      yield {'mention_sentence': mention_sentence,
             'label': label,
             'page_token_cnts': page_token_cnts,
             'p_prior': p_prior,
             'candidate_ids': candidate_ids.tolist(),
             'candidate_mention_sim': candidate_mention_sim}
