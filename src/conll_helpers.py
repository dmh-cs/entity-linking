import re
import pydash as _
from parsers import parse_for_tokens, parse_for_sentence_spans

def _get_doc_lines(lines):
  divs = [i for i, line in enumerate(lines) if '-DOCSTART-' in line] + [len(lines)]
  return [lines[start + 1 : end] for start, end in zip(divs, divs[1:])]

def get_documents(lines):
  doc_lines = _get_doc_lines(lines)
  return [' '.join([line.split('\t')[0]
                    if len(line.split('\t')) != 0 else '\n' for line in doc])
          for doc in doc_lines]

def get_mentions(lines):
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

def _get_mention_sentence(doc, mention, seek, span):
  mention_start_seek_offset = _.index_of(doc[seek:], mention)
  mention_start_sentence_offset = seek - span[0] + mention_start_seek_offset
  to_idx = mention_start_sentence_offset + len(mention)
  sentence = doc[span[0]:span[1]]
  return (parse_for_tokens(sentence), span[0] + to_idx)

def _create_span(spans, mention_start_idx, mention_end_idx):
  start_span_idx = _.find_index(spans,
                                lambda span: span[0] <= mention_start_idx and span[1] >= mention_start_idx)
  assert start_span_idx != -1
  end_span_offset = _.find_index(spans[start_span_idx:],
                                 lambda span: mention_end_idx <= span[1] and mention_end_idx >= span[0])
  assert end_span_offset != -1
  end_span_idx = start_span_idx + end_span_offset
  return spans[start_span_idx][0], spans[end_span_idx][1]

def get_splits(documents, mentions):
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

def get_mention_sentences(documents, mentions):
  all_sentences = []
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
      sentence, seek = _get_mention_sentence(doc, mention, seek, span)
      all_sentences.append(sentence)
      mention_idx += 1
  return all_sentences

def get_entity_page_ids(lines):
  doc_lines = _get_doc_lines(lines)
  return [int(line.split('\t')[5])
          for doc in doc_lines
          for line in doc
          if len(line.split('\t')) >= 5 and line.split('\t')[1] == 'B']

def from_page_ids_to_entity_ids(cursor, page_ids):
  cursor.execute('select entity_id, p.source_id from entity_by_page e join pages p on e.`page_id` = p.id where p.source_id in (' + str(page_ids)[1:-1] + ')')
  lookup = {row['source_id']: row['entity_id']
            for row in cursor.fetchall() if row is not None}
  return [lookup[page_id] if page_id in lookup else -1 for page_id in page_ids]

def get_doc_id_per_mention(lines):
  doc_lines = _get_doc_lines(lines)
  return [doc_id
          for doc_id, doc in enumerate(doc_lines)
          for line in doc if len(line.split('\t')) >= 5 and line.split('\t')[1] == 'B']

def get_mentions_by_doc_id(lines):
  doc_lines = _get_doc_lines(lines)
  return [[line.split('\t')[2]
           for line in doc if len(line.split('\t')) >= 5 and line.split('\t')[1] == 'B']
          for doc in doc_lines]
