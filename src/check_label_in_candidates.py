import os
import sys

from dotenv import load_dotenv
from torch.utils.data import DataLoader
from pyrsistent import m
import pydash as _
import torch

from runner import Runner
from data_fetchers import get_connection

def collate(batch):
  return {'label': torch.tensor([sample['label'] for sample in batch]),
          'candidates': torch.stack([sample['candidates'] for sample in batch])}

def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  load_dotenv(dotenv_path='.env')
  paths = m(lookups=os.getenv("LOOKUPS_PATH"),
            page_id_order=os.getenv("PAGE_ID_ORDER_PATH"))
  runner = Runner(device, paths=paths)
  runner.load_caches()
  runner.init_entity_embeds()
  db_connection = get_connection()
  with db_connection.cursor() as cursor:
    dataloader = DataLoader(dataset=runner._get_dataset(cursor, is_test=True),
                            batch_sampler=runner._get_sampler(cursor, is_test=True),
                            collate_fn=collate)
    acc = 0
    n = 0
    for batch in dataloader:
      for label, candidates in zip(batch['label'], batch['candidates']):
        if int(label) in candidates.tolist():
          acc += 1
      n += 1
      print(acc, n)
    print(acc, n)


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
