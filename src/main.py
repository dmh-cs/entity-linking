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
  args = getopt.getopt(_.tail(sys.argv), '', ['load_model', 'model_path=', 'name='])[0]
  flags = [_.head(arg) for arg in args]
  model_path_pair = _.find(args, lambda pair: 'model_path' in pair[0])
  name_pair = _.find(args, lambda pair: 'name' in pair[0])
  train_params = m()
  run_params = m(load_model='--load_model' in flags)
  model_params = m()
  paths = m(lookups=os.getenv("LOOKUPS_PATH"),
            page_id_order=os.getenv("PAGE_ID_ORDER_PATH"))
  if model_path_pair:
    paths = paths.set('model', model_path_pair[1])
  name = name_pair[1] if name_pair else ''
  runner = Runner(device=device,
                  paths=paths,
                  train_params=train_params,
                  model_params=model_params,
                  run_params=run_params,
                  name=name)
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
