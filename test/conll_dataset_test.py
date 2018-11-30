from collections import defaultdict

from conll_dataset import _get_documents, _get_mentions, _get_splits, _get_entity_page_ids, _get_doc_id_per_mention, _get_mentions_by_doc_id

class Vocab(defaultdict):
  def __missing__(self, key):
    self[key] = len(self)
    return self[key]

def test__get_documents():
  with open('./test/fixtures/conll', 'r') as fh:
    lines = fh.read().strip().split('\n')
  documents = _get_documents(lines)
  assert documents == ["EU rejects German call to boycott British lamb .  Peter Blackburn  BRUSSELS 1996-08-22  The European Commission said on Thursday it disagreed with German advice to consumers to shun British lamb until scientists determine whether mad cow disease can be transmitted to sheep .  Germany 's representative to the European Union 's veterinary committee Werner Zwingmann said on Wednesday consumers should buy sheepmeat from countries other than Britain until the scientific advice was clearer . ",
                       'EU rejects German call to boycott British lamb .']

def test__get_mentions():
  with open('./test/fixtures/conll', 'r') as fh:
    lines = fh.read().strip().split('\n')
  mentions = _get_mentions(lines)
  assert mentions == ['German', 'British', 'BRUSSELS', 'European Commission', 'German', 'British', 'Germany', 'European Union', 'Britain', 'German', 'British']

def test__get_entity_page_ids():
  with open('./test/fixtures/conll', 'r') as fh:
    lines = fh.read().strip().split('\n')
  page_ids = _get_entity_page_ids(lines)
  assert page_ids == [11867, 31717, 3708, 9974, 11867, 31717, 11867, 9317, 31717, 11867, 31717]

def test__get_doc_id_per_mention():
  with open('./test/fixtures/conll', 'r') as fh:
    lines = fh.read().strip().split('\n')
  doc_id_per_mention = _get_doc_id_per_mention(lines)
  assert doc_id_per_mention == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]

def test__get_mentions_by_doc_id():
  with open('./test/fixtures/conll', 'r') as fh:
    lines = fh.read().strip().split('\n')
  mentions_by_doc_id = _get_mentions_by_doc_id(lines)
  assert mentions_by_doc_id == [['German', 'British', 'BRUSSELS', 'European Commission', 'German', 'British', 'Germany', 'European Union', 'Britain'],
                                ['German', 'British']]

def test__get_splits():
  with open('./test/fixtures/conll', 'r') as fh:
    lines = fh.read().strip().split('\n')
  documents = _get_documents(lines)
  mentions = _get_mentions(lines)
  splits = _get_splits(documents, mentions)
  assert splits == [[['EU', 'rejects', 'German'],
                     ['German', 'call', 'to', 'boycott', 'British', 'lamb', '.']],
                    [['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British'],
                     ['British', 'lamb', '.']],
                    [['Peter', 'Blackburn', 'BRUSSELS'],
                     ['BRUSSELS', '1996', '-', '08', '-', '22', 'The', 'European', 'Commission', 'said', 'on', 'Thursday', 'it', 'disagreed', 'with', 'German', 'advice', 'to', 'consumers', 'to', 'shun', 'British', 'lamb', 'until', 'scientists', 'determine', 'whether', 'mad', 'cow', 'disease', 'can', 'be', 'transmitted', 'to', 'sheep', '.']],
                    [['Peter', 'Blackburn', 'BRUSSELS', '1996', '-', '08', '-', '22', 'The', 'European', 'Commission'],
                     ['European', 'Commission', 'said', 'on', 'Thursday', 'it', 'disagreed', 'with', 'German', 'advice', 'to', 'consumers', 'to', 'shun', 'British', 'lamb', 'until', 'scientists', 'determine', 'whether', 'mad', 'cow', 'disease', 'can', 'be', 'transmitted', 'to', 'sheep', '.']],
                    [['Peter', 'Blackburn', 'BRUSSELS', '1996', '-', '08', '-', '22', 'The', 'European', 'Commission', 'said', 'on', 'Thursday', 'it', 'disagreed', 'with', 'German'],
                     ['German', 'advice', 'to', 'consumers', 'to', 'shun', 'British', 'lamb', 'until', 'scientists', 'determine', 'whether', 'mad', 'cow', 'disease', 'can', 'be', 'transmitted', 'to', 'sheep', '.']],
                    [['Peter', 'Blackburn', 'BRUSSELS', '1996', '-', '08', '-', '22', 'The', 'European', 'Commission', 'said', 'on', 'Thursday', 'it', 'disagreed', 'with', 'German', 'advice', 'to', 'consumers', 'to', 'shun', 'British'],
                     ['British', 'lamb', 'until', 'scientists', 'determine', 'whether', 'mad', 'cow', 'disease', 'can', 'be', 'transmitted', 'to', 'sheep', '.']],
                    [['Germany'], ['Germany', "'s", 'representative', 'to', 'the', 'European', 'Union', "'s", 'veterinary', 'committee', 'Werner', 'Zwingmann', 'said', 'on', 'Wednesday', 'consumers', 'should', 'buy', 'sheepmeat', 'from', 'countries', 'other', 'than', 'Britain', 'until', 'the', 'scientific', 'advice', 'was', 'clearer', '.']],
                    [['Germany', "'s", 'representative', 'to', 'the', 'European', 'Union'], ['European', 'Union', "'s", 'veterinary', 'committee', 'Werner', 'Zwingmann', 'said', 'on', 'Wednesday', 'consumers', 'should', 'buy', 'sheepmeat', 'from', 'countries', 'other', 'than', 'Britain', 'until', 'the', 'scientific', 'advice', 'was', 'clearer', '.']],
                    [['Germany', "'s", 'representative', 'to', 'the', 'European', 'Union', "'s", 'veterinary', 'committee', 'Werner', 'Zwingmann', 'said', 'on', 'Wednesday', 'consumers', 'should', 'buy', 'sheepmeat', 'from', 'countries', 'other', 'than', 'Britain'], ['Britain', 'until', 'the', 'scientific', 'advice', 'was', 'clearer', '.']],
                    [['EU', 'rejects', 'German'],
                     ['German', 'call', 'to', 'boycott', 'British', 'lamb', '.']],
                    [['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British'],
                     ['British', 'lamb', '.']]]
