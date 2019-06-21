from operator import itemgetter
from itertools import groupby

import pydash as _
import xgboost as xgb
from xgboost import DMatrix
from sklearn.datasets import load_svmlight_file
from rabbit_ml import get_cli_args

from utils import items_to_str
from args_config import args

def main():
  p = get_cli_args(args)
  x_train, y_train, qid_train = load_svmlight_file(p.train.xgboost_train_path, query_id=True) # pylint: disable=unbalanced-tuple-unpacking
  x_test, y_test, qid_test = load_svmlight_file(p.train.xgboost_test_path, query_id=True) # pylint: disable=unbalanced-tuple-unpacking
  train_dmatrix = DMatrix(x_train, y_train)
  test_dmatrix = DMatrix(x_test, y_test)
  train_dmatrix.set_group([len(list(g)) for __, g in groupby(qid_train)])
  test_dmatrix.set_group([len(list(g)) for __, g in groupby(qid_test)])
  params = {'objective': 'rank:pairwise',
            'eval_metric': ['error', 'map@1'],
            'eta': 0.1,
            'gamma': 1.0,
            'min_child_weight': 0.1,
            'max_depth': 6}
  xgb_model = xgb.train(params, train_dmatrix, num_boost_round=100,
                        evals=[(test_dmatrix, 'validation')])
  xgb_train_str = items_to_str(_.omit(params, 'objective', 'eval_metric').items(),
                               sort_by=itemgetter(0))
  xgb_model.save_model(xgb_train_str + '_model.xgb')


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
