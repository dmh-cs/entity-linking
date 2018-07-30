import getopt
import os
import sys

from dotenv import load_dotenv
from pyrsistent import m
import pydash as _
import torch

from runner import Runner

args_with_values = [{'name': 'batch_size', 'for': 'train_param', 'type': int},
                    {'name': 'num_epochs', 'for': 'train_param', 'type': int},
                    {'name': 'train_size', 'for': 'train_param', 'type': int},
                    {'name': 'dropout_keep_prob', 'for': 'train_param', 'type': float},
                    {'name': 'embed_len', 'for': 'model_param', 'type': int},
                    {'name': 'word_embed_len', 'for': 'model_param', 'type': int},
                    {'name': 'num_candidates', 'for': 'model_param', 'type': int},
                    {'name': 'local_encoder_lstm_size', 'for': 'model_param', 'type': int},
                    {'name': 'document_encoder_lstm_size', 'for': 'model_param', 'type': int},
                    {'name': 'num_lstm_layers', 'for': 'model_param', 'type': int},
                    {'name': 'word_embedding_set', 'for': 'model_param', 'type': str},
                    {'name': 'ablation', 'for': 'model_param', 'type': lambda string: string.split(',')}]

def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  load_dotenv(dotenv_path='.env')
  args = getopt.getopt(_.tail(sys.argv), '', ['load_model'] + [arg['name'] + '=' for arg in args_with_values])[0]
  flags = [_.head(arg) for arg in args]
  train_params = m()
  run_params = m(load_model='--load_model' in flags)
  model_params = m()
  paths = m(lookups=os.getenv("LOOKUPS_PATH"),
            page_id_order=os.getenv("PAGE_ID_ORDER_PATH"))
  for arg in args_with_values:
    name = arg['name']
    pair = _.find(args, lambda pair: name in pair[0])
    if pair:
      parsed = arg['type'](pair[1])
      if arg['for'] == 'path':
        paths = paths.set(name, parsed)
      elif arg['for'] == 'model_param':
        model_params = model_params.set(name, parsed)
      elif arg['for'] == 'train_param':
        train_params = train_params.set(name, parsed)
  name_pair = _.find(args, lambda pair: 'name' in pair[0])
  name = name_pair[1] if name_pair else ''
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
