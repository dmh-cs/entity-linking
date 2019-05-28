from collections import Counter
import json
import wikipedia
from progressbar import progressbar

import numpy as np
from nltk.stem.snowball import SnowballStemmer

from simple_conll_dataset import SimpleCoNLLDataset
from parsers import parse_text_for_tokens
from data_fetchers import get_connection

env = '.env'

def get_desc_fs(cursor, stemmer, cand_id):
  try:
    cursor.execute(f'select page_id from entity_by_page where entity_id = {cand_id}')
    page_id = cursor.fetchone()['page_id']
    cursor.execute(f'select content from pages where id = {page_id}')
    page_content = cursor.fetchone()['content']
  except:
    cursor.execute(f'select text from entities where id = {cand_id}')
    entity_text = cursor.fetchone()['text']
    try:
      page = wikipedia.page(entity_text)
      page_content = page.content
    except:
      page_content = ''
  tokenized = parse_text_for_tokens(page_content)
  return dict(Counter(stemmer.stem(token) for token in tokenized))

def get_test_set(cursor, lookups_path, train_size, use_custom=False):
  conll_path = 'custom.tsv' if use_custom else './AIDA-YAGO2-dataset.tsv'
  return SimpleCoNLLDataset(cursor, conll_path, lookups_path, train_size)

def get_stemmed_f(stemmer, tokens):
  return dict(Counter(stemmer.stem(token) for token in tokens))

def load_idf():
  with open('./wiki_idf.json') as fh:
    return json.load(fh)

def main():
  lookups_path = '../wp-preprocessing-el/lookups.pkl_local'
  train_size = 0.8
  num_correct = 0
  missed_idxs = []
  guessed_when_missed = []
  idf = load_idf()
  stemmer = SnowballStemmer('english')
  db_connection = get_connection(env)
  with db_connection.cursor() as cursor:
    conll_test_set = get_test_set(cursor, lookups_path, train_size)
    for idx, row in progressbar(enumerate(conll_test_set)):
      mention_f = get_stemmed_f(stemmer, row['mention_sentence'])
      cand_scores = []
      for cand_id in row['candidate_ids']:
        desc_f = get_desc_fs(cursor, stemmer, cand_id)
        cand_scores.append(sum(cnt * desc_f.get(token, 0) * idf.get(token, 1)
                               for token, cnt in mention_f.items()))
      if len(cand_scores) == 0:
        guess = 0
      else:
        guess = row['candidate_ids'][np.argmax(cand_scores)]
      if guess == row['label']:
        num_correct += 1
      else:
        missed_idxs.append(idx)
        guessed_when_missed.append(guess)
    print(num_correct / len(conll_test_set))
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
