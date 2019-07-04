import os

import pickle
import pymysql.cursors
from dotenv import load_dotenv
from progressbar import progressbar

from simple_mention_dataset import SimpleMentionDataset
from simple_conll_dataset import SimpleCoNLLDataset

from rabbit_ml import get_cli_args

from args_config import args

def main():
  p = get_cli_args(args)
  with open('./tokens.pkl', 'rb') as fh: token_idx_lookup = pickle.load(fh)
  with open('./glove_token_idx_lookup.pkl', 'rb') as fh: full_token_idx_lookup = pickle.load(fh)
  load_dotenv(dotenv_path=p.run.env_path)
  EL_DATABASE_NAME = os.getenv("DBNAME")
  DATABASE_USER = os.getenv("DBUSER")
  DATABASE_PASSWORD = os.getenv("DBPASS")
  DATABASE_HOST = os.getenv("DBHOST")
  with open(p.train.page_id_order_path, 'rb') as fh:
    page_id_order = pickle.load(fh)
  page_ids = page_id_order[:p.train.num_pages_to_use]
  connection = pymysql.connect(host=DATABASE_HOST,
                               user=DATABASE_USER,
                               password=DATABASE_PASSWORD,
                               db=EL_DATABASE_NAME,
                               charset='utf8mb4',
                               use_unicode=True,
                               cursorclass=pymysql.cursors.DictCursor)
  with connection.cursor() as cursor:
    cursor.execute("SET NAMES utf8mb4;")
    cursor.execute("SET CHARACTER SET utf8mb4;")
    cursor.execute("SET character_set_connection=utf8mb4;")

    if p.train.train_on_conll:
      conll_path = 'custom.tsv' if p.run.use_custom else './AIDA-YAGO2-dataset.tsv'
      dataset = SimpleCoNLLDataset(cursor,
                                   token_idx_lookup,
                                   full_token_idx_lookup,
                                   conll_path,
                                   p.run.lookups_path,
                                   p.run.idf_path,
                                   p.train.train_size)
    else:
      dataset = SimpleMentionDataset(cursor,
                                     token_idx_lookup,
                                     full_token_idx_lookup,
                                     page_ids,
                                     p.run.lookups_path,
                                     p.run.idf_path,
                                     p.train.train_size)
    train_str = '_'.join(['conll' if p.train.train_on_conll else 'wiki',
                          'custom' if p.run.use_custom else '',
                          str(p.train.num_pages_to_use)])
    with open('./4data_{}'.format(train_str), 'w') as fh:
      for item_num, item in progressbar(enumerate(dataset)):
        fh.write('{}\n'.format(str(item)))
        if item_num % 1000 == 0: fh.flush()



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
