import pickle
from random import shuffle

import pydash as _

from data_fetchers import get_connection

def get_page_id_order(cursor):
  cursor.execute('select id from pages')
  page_ids = []
  while True:
    results = cursor.fetchmany(10000)
    if _.is_empty(results): break
    page_ids.extend([row['id'] for row in results])
  shuffle(page_ids)
  return page_ids

def main():
  db_connection = get_connection()
  with db_connection.cursor() as cursor:
    page_id_order = get_page_id_order(cursor)
  with open('page_id_order.pkl', 'wb+') as f:
    pickle.dump(page_id_order, f)

if __name__ == "__main__": main()
