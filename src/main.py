import os

from dotenv import load_dotenv
from pyrsistent import m
import pydash as _
import torch

from runner import Runner

def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  load_dotenv(dotenv_path='.env')
  train_params = m(load_model=False)
  model_params = m()
  paths = m(lookups_path=os.getenv("LOOKUPS_PATH"),
            page_id_order_path=os.getenv("PAGE_ID_ORDER_PATH"))
  runner = Runner(device=device,
                  paths=paths,
                  train_params=train_params,
                  model_params=model_params)
  runner.run()


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
