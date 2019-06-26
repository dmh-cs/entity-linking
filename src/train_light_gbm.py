from operator import itemgetter
from itertools import groupby

import pydash as _
import xgboost as xgb
import numpy as np
from xgboost import DMatrix
from sklearn.datasets import load_svmlight_file
from rabbit_ml import get_cli_args

from utils import items_to_str
from args_config import args

import lightgbm as lgb

def main():
  p = get_cli_args(args)
  try:
    open('train.bin').close()
    open('eval.bin').close()
    lgb_train = lgb.Dataset('train.bin')
    lgb_eval = lgb.Dataset('eval.bin', reference=lgb_train)
  except:
    x_train, y_train, qid_train = load_svmlight_file(p.train.xgboost_train_path, query_id=True) # pylint: disable=unbalanced-tuple-unpacking
    x_test, y_test, qid_test = load_svmlight_file(p.train.xgboost_test_path, query_id=True) # pylint: disable=unbalanced-tuple-unpacking
    x_train = x_train.todense()
    x_test = x_test.todense()
    lgb_train = lgb.Dataset(np.array(x_train),
                            np.array(y_train.squeeze()),
                            group=[len(list(g)) for __, g in groupby(qid_train)])
    lgb_eval = lgb.Dataset(np.array(x_test),
                           np.array(y_test.squeeze()),
                           reference=lgb_train,
                           group=[len(list(g)) for __, g in groupby(qid_test)])
    lgb_train.save_binary("train.bin")
    lgb_eval.save_binary("eval.bin")

  params = {
    'boosting_type': 'gbdt',
    'objective': 'lambdarank',
    'metric': {'ndcg'},
    'ndcg_eval_at': [1],
    'metric_freq': 1,
    'max_bin': 255,
    'num_trees': 100,
    'num_leaves': 100,
    'learning_rate': 0.1,
    'num_iterations': 300,
    'num_threads': 8,
    'feature_fraction': 1.0,
    'bagging_fraction': 0.9,
    'bagging_freq': 1,
    'verbose': 0,
  }
  gbm = lgb.train(params,
                  lgb_train,
                  num_boost_round=100,
                  valid_sets=lgb_eval)
  xgb_train_str = items_to_str(_.omit(params, 'objective', 'eval_metric').items(),
                               sort_by=itemgetter(0))
  gbm.save_model('model' + xgb_train_str + '.light')


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
