import ast

from rabbit_ml import get_cli_args
from progressbar import progressbar
import pydash as _

from args_config import args

def libsvm_format_row(row_label, query_id, row_features):
  row_rank = 1 if row_label == 0 else 2
  row = ''
  row += '{} '.format(row_rank)
  row += 'qid:{} '.format(query_id)
  row += ' '.join(':'.join(str(elem) for elem in pair)
                  for pair in enumerate(row_features, 1))
  return row

def main():
  p = get_cli_args(args)
  with open(p.run.xgboost_path, 'w') as data_fh:
    with open(p.run.txt_dataset_path) as read_fh:
      dataset = [ast.literal_eval(line) for line in read_fh.readlines()]
      features, candidate_ids, labels = zip(*dataset)
      iterator = progressbar(enumerate(zip(features, candidate_ids, labels), 1))
      for mention_num, mention_info in iterator:
        mention_features, mention_candidate_ids, label = mention_info
        for candidate_features, candidate_id in zip(mention_features, mention_candidate_ids):
          row_label = 1 if candidate_id == label else 0
          row_features = candidate_features
          row = libsvm_format_row(row_label, mention_num, row_features)
          data_fh.write('{}\n'.format(row))



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
