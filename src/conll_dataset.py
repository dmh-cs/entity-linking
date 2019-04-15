import Levenshtein
import re

import numpy as np
import torch
from torch.utils.data import Dataset

import pydash as _

from data_transformers import embed_page_content, get_bag_of_nouns
from data_fetchers import get_candidate_ids, get_p_prior, get_candidate_strs
from parsers import parse_for_sentence_spans, parse_for_tokens

def _get_doc_lines(lines):
  divs = [i for i, line in enumerate(lines) if '-DOCSTART-' in line] + [len(lines)]
  return [lines[start + 1 : end] for start, end in zip(divs, divs[1:])]

def _get_documents(lines):
  doc_lines = _get_doc_lines(lines)
  return [' '.join([line.split('\t')[0]
                    if len(line.split('\t')) != 0 else '\n' for line in doc])
          for doc in doc_lines]

def _get_mentions(lines):
  doc_lines = _get_doc_lines(lines)
  return [line.split('\t')[2]
          for doc in doc_lines
          for line in doc
          if len(line.split('\t')) >= 5 and line.split('\t')[1] == 'B']

def _get_mention_splits(doc, mention, seek, span):
  mention_start_seek_offset = _.index_of(doc[seek:], mention)
  mention_start_sentence_offset = seek - span[0] + mention_start_seek_offset
  to_idx = mention_start_sentence_offset + len(mention)
  sentence = doc[span[0]:span[1]]
  return ([parse_for_tokens(sentence[:mention_start_sentence_offset] + mention),
           parse_for_tokens(mention + sentence[to_idx:])],
          span[0] + to_idx)

def _create_span(spans, mention_start_idx, mention_end_idx):
  start_span_idx = _.find_index(spans,
                                lambda span: span[0] <= mention_start_idx and span[1] >= mention_start_idx)
  assert start_span_idx != -1
  end_span_offset = _.find_index(spans[start_span_idx:],
                                 lambda span: mention_end_idx <= span[1] and mention_end_idx >= span[0])
  assert end_span_offset != -1
  end_span_idx = start_span_idx + end_span_offset
  return spans[start_span_idx][0], spans[end_span_idx][1]

def _get_splits(documents, mentions):
  all_splits = []
  doc_sentence_spans = [parse_for_sentence_spans(doc) for doc in documents]
  mention_idx = 0
  for doc, spans in zip(documents, doc_sentence_spans):
    seek = 0
    while mention_idx < len(mentions):
      mention = mentions[mention_idx]
      mention_start_offset = _.index_of(doc[seek:], mention)
      if mention_start_offset == -1:
        mention_start_offset = _.index_of(doc[seek:], re.sub(' +', ' ', ' , '.join(' . '.join(mention.split('.')).split(','))).replace('D . C .', 'D.C.'))
        if mention_start_offset == -1: break
      mention_start_idx = mention_start_offset + seek
      mention_end_idx = mention_start_idx + len(mention)
      span = _create_span(spans, mention_start_idx, mention_end_idx)
      splits, seek = _get_mention_splits(doc, mention, seek, span)
      all_splits.append(splits)
      mention_idx += 1
  return all_splits

def _get_entity_page_ids(lines):
  doc_lines = _get_doc_lines(lines)
  return [int(line.split('\t')[5])
          for doc in doc_lines
          for line in doc
          if len(line.split('\t')) >= 5 and line.split('\t')[1] == 'B']

def _from_page_ids_to_entity_ids(cursor, page_ids):
  cursor.execute('select entity_id, p.source_id from entity_by_page e join pages p on e.`page_id` = p.id where p.source_id in (' + str(page_ids)[1:-1] + ')')
  lookup = {row['source_id']: row['entity_id']
            for row in cursor.fetchall() if row is not None}
  return [lookup[page_id] if page_id in lookup else -1 for page_id in page_ids]

def _get_doc_id_per_mention(lines):
  doc_lines = _get_doc_lines(lines)
  return [doc_id
          for doc_id, doc in enumerate(doc_lines)
          for line in doc if len(line.split('\t')) >= 5 and line.split('\t')[1] == 'B']

def _get_mentions_by_doc_id(lines):
  doc_lines = _get_doc_lines(lines)
  return [[line.split('\t')[2]
           for line in doc if len(line.split('\t')) >= 5 and line.split('\t')[1] == 'B']
          for doc in doc_lines]

class CoNLLDataset(Dataset):
  def __init__(self,
               cursor,
               entity_candidates_prior,
               embedding,
               token_idx_lookup,
               num_entities,
               num_candidates,
               entity_label_lookup,
               path='./AIDA-YAGO2-dataset.tsv',
               use_wiki2vec=False):
    self.cursor = cursor
    self.entity_candidates_prior = entity_candidates_prior
    self.embedding = embedding
    self.token_idx_lookup = token_idx_lookup
    self.num_entities = num_entities
    self.num_candidates = num_candidates
    with open(path, 'r') as fh:
      self.lines = fh.read().strip().split('\n')[:-1]
    self.documents = _get_documents(self.lines)
    self.embedded_documents = [embed_page_content(self.embedding, self.token_idx_lookup, document)
                               for document in self.documents]
    self.mentions = _get_mentions(self.lines)
    self.sentence_splits = _get_splits(self.documents, self.mentions)
    self.entity_page_ids = _get_entity_page_ids(self.lines)
    self.labels = _from_page_ids_to_entity_ids(cursor, self.entity_page_ids)
    self.with_label = [i for i, x in enumerate(self.labels) if x != -1]
    self.mention_doc_id = _get_doc_id_per_mention(self.lines)
    self.mentions_by_doc_id = _get_mentions_by_doc_id(self.lines)
    self.entity_label_lookup = entity_label_lookup
    self.entity_id_lookup = {int(label): entity_id for entity_id, label in self.entity_label_lookup.items()}
    self.use_wiki2vec = use_wiki2vec

  def __len__(self):
    return len(self.with_label)

  def __getitem__(self, idx):
    if self.use_wiki2vec:
      return self._getitem_wiki2vec(idx)
    else:
      self._getitem_deep_el(idx)

  def _getitem_deep_el(self, idx):
    idx = self.with_label[idx]
    label = self.entity_label_lookup.get(self.labels[idx]) or -1
    mention = self.mentions[idx]
    candidate_ids = get_candidate_ids(self.entity_candidates_prior,
                                      self.num_entities,
                                      self.num_candidates,
                                      mention,
                                      label)
    candidates = get_candidate_strs(self.cursor, [self.entity_id_lookup[cand_id] for cand_id in candidate_ids.tolist()])
    return {'sentence_splits': self.sentence_splits[idx],
            'label': label,
            'embedded_page_content': self.embedded_documents[self.mention_doc_id[idx]],
            'entity_page_mentions': embed_page_content(self.embedding,
                                                       self.token_idx_lookup,
                                                       ' '.join(self.mentions_by_doc_id[self.mention_doc_id[idx]])),
            'p_prior': get_p_prior(self.entity_candidates_prior, mention, candidate_ids),
            'candidate_ids': candidate_ids,
            'candidate_mention_sim': torch.tensor([Levenshtein.ratio(mention, candidate)
                                                   for candidate in candidates])}

  def _getitem_wiki2vec(self, idx):
    idx = self.with_label[idx]
    label = self.entity_label_lookup.get(self.labels[idx]) or -1
    mention = self.mentions[idx]
    candidate_ids = get_candidate_ids(self.entity_candidates_prior,
                                      self.num_entities,
                                      self.num_candidates,
                                      mention,
                                      label)
    bag_of_nouns = get_bag_of_nouns(self.documents[self.mention_doc_id[idx]])
    candidates = get_candidate_strs(self.cursor, [self.entity_id_lookup[cand_id] for cand_id in candidate_ids.tolist()])
    return {'label': label,
            'bag_of_nouns': bag_of_nouns,
            'p_prior': get_p_prior(self.entity_candidates_prior, mention, candidate_ids),
            'candidate_ids': candidate_ids,
            'candidate_mention_sim': torch.tensor([Levenshtein.ratio(mention, candidate)
                                                   for candidate in candidates])}
