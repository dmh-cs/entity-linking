import getopt
import os
import sys

from dotenv import load_dotenv
from pyrsistent import m
import pydash as _
import torch

from runner import Runner

def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  load_dotenv(dotenv_path='.env')
  flags = [_.head(arg) for arg in getopt.getopt(_.tail(sys.argv), '', ['load_model'])[0]]
  load_model = '--load_model' in flags
  train_params = m(load_model=load_model)
  run_params = m(load_model=load_model)
  model_params = m()
  paths = m(lookups=os.getenv("LOOKUPS_PATH"),
            page_id_order=os.getenv("PAGE_ID_ORDER_PATH"))
  runner = Runner(device=device,
                  paths=paths,
                  train_params=train_params,
                  model_params=model_params,
                  run_params=run_params)
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
