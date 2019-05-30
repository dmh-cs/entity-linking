from collections import Counter
import json
import wikipedia
from progressbar import progressbar
import sys
import requests
from pymongo import MongoClient
import string

import numpy as np
from nltk.stem.snowball import SnowballStemmer

from simple_conll_dataset import SimpleCoNLLDataset
from parsers import parse_text_for_tokens
from data_fetchers import get_connection


def get_desc_fs(pages_db, cursor, stemmer, cand_ids):
  cursor.execute('select id, text from entities where id in (' + str(cand_ids)[1:-1] + ')')
  entity_text_lookup = {row['id']: row['text'] for row in cursor.fetchall()}
  fs = {}
  for entity_id, entity_text in entity_text_lookup.items():
    try:
      # page = wikipedia.page(entity_text)
      # page_content = page.content[:2000]
      # response = requests.get('https://en.wikipedia.org/w/api.php',params={'action': 'query', 'format': 'json', 'titles': entity_text, 'prop': 'extracts', 'redirects': True, 'exintro': True, 'explaintext': True,}).json()
      # page_content = next(iter(response['query']['pages'].values()))
      page_content = pages_db.find_one({'_id': entity_text})['plaintext'][:2000]
    except:
      page_content = ''
    tokenized = parse_text_for_tokens(page_content)
    fs[entity_id] = dict(Counter(stemmer.stem(token) for token in tokenized))
  return fs

def get_test_set(cursor, lookups_path, train_size, use_custom=False):
  conll_path = 'custom.tsv' if use_custom else './AIDA-YAGO2-dataset.tsv'
  return SimpleCoNLLDataset(cursor, conll_path, lookups_path, train_size)

def get_stemmed_f(stemmer, tokens):
  return dict(Counter(stemmer.stem(token) for token in tokens))

def load_idf():
  # with open('./wiki_idf.json') as fh:
  with open('./wiki_idf_stem.json') as fh:
    return json.load(fh)

def main():
  client = MongoClient()
  dbname = 'enwiki'
  db = client[dbname]
  pages_db = db['pages']
  use_custom = '--use_custom' in sys.argv
  if '--remote' in sys.argv:
    env = '.env_remote'
    lookups_path = '../wp-preprocessing-el/lookups.pkl'
    train_size = 1.0
  else:
    env = '.env'
    lookups_path = '../wp-preprocessing-el/lookups.pkl_local'
    train_size = 0.8
  num_correct = 0
  total_with_entity_id = 0
  missed_idxs = []
  guessed_when_missed = []
  all_scores = []
  all_candidates = []
  idf = load_idf()
  stemmer = SnowballStemmer('english')
  db_connection = get_connection(env)
  with db_connection.cursor() as cursor:
    conll_test_set = get_test_set(cursor, lookups_path, train_size, use_custom=use_custom)
    for idx, row in progressbar(enumerate(conll_test_set)):
      mention_f = get_stemmed_f(stemmer, row['mention_sentence'])
      cand_scores = []
      if len(row['candidate_ids']) == 0:
        guess = 0
      else:
        desc_fs = get_desc_fs(pages_db, cursor, stemmer, row['candidate_ids'])
        for cand_id in row['candidate_ids']:
          desc_f = desc_fs[cand_id]
          cand_scores.append(sum(cnt * desc_f.get(token, 0) * idf.get(token,
                                                                      idf.get(token.lower(), 0.0))
                                 for token, cnt in mention_f.items()))
        guess = row['candidate_ids'][np.argmax(cand_scores)]
      all_scores.append(cand_scores)
      all_candidates.append(row['candidate_strs'])
      if guess == row['label']:
        num_correct += 1
      else:
        missed_idxs.append(idx)
        guessed_when_missed.append(guess)
      if (row['label'] in row['candidate_ids']) and (len(desc_fs[row['label']]) != 0):
        total_with_entity_id += 1
    import ipdb; ipdb.set_trace()



if __name__ == "__main__":
  import ipdb
  import traceback
  import sys

  try:
    main()
  except: # pylint: disable=bare-except
    extype, value, tb = sys.exc_info()
    traceback.print_exc()
    ipdb.post_mortem(tb)
