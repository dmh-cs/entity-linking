import os

import pickle
import pymysql.cursors
from dotenv import load_dotenv
import torch
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.data import DataLoader

from utils import tensors_to_device, to_idx
from ltr_bow import LtRBoW
from simple_mention_dataset import SimpleMentionDataset, collate_simple_mention

def main():
  model = LtRBoW()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  optimizer = optim.Adam(model.parameters())
  num_epochs = 5
  batch_size = 100
  num_pages_to_use = 10000
  load_dotenv(dotenv_path='.env')
  page_id_order_path = '../wp-entity-preprocessing/page_id_order.pkl_local'
  lookups_path = '../wp-preprocessing-el/lookups.pkl_local'
  idf_path = './wiki_idf_stem.json'
  train_size = 0.8
  EL_DATABASE_NAME = os.getenv("EL_DBNAME")
  DATABASE_USER = os.getenv("DBUSER")
  DATABASE_PASSWORD = os.getenv("DBPASS")
  DATABASE_HOST = os.getenv("DBHOST")
  with open(page_id_order_path, 'b') as fh:
    page_id_order = pickle.load(fh)
  page_ids = page_id_order[:num_pages_to_use]
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
    dataset = SimpleMentionDataset(cursor, page_ids, lookups_path, idf_path, train_size)
    dataloader = DataLoader(dataset,
                            batch_sampler=BatchSampler(RandomSampler(dataset), batch_size, False),
                            collate_fn=collate_simple_mention)
    for epoch_num in range(num_epochs):
      for batch_num, batch in enumerate(dataloader):
        model.train()
        optimizer.zero_grad()
        batch = tensors_to_device(batch, device)
        labels = _get_labels_for_batch(batch['label'], batch['candidate_ids'])
        scores = [scores[(labels != -1).nonzero().reshape(-1)] for scores in scores]
        labels = labels[(labels != -1).nonzero().reshape(-1)]
        loss = calc_loss(scores, labels)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
          model.eval()
          error = _classification_error(probas, labels)
      torch.save(model.state_dict(), './ltr_model')



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
