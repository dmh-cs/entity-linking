import pyltr

import pydash as _
from sklearn.datasets import load_svmlight_file
from rabbit_ml import get_cli_args

from args_config import args

def main():
  p = get_cli_args(args)
  x_train, y_train, qid_train = load_svmlight_file(p.train.xgboost_train_path, query_id=True) # pylint: disable=unbalanced-tuple-unpacking
  x_test, y_test, qid_test = load_svmlight_file(p.train.xgboost_test_path, query_id=True) # pylint: disable=unbalanced-tuple-unpacking
  metric = pyltr.metrics.AP(k=1)
  model = pyltr.models.LambdaMART(
      metric=metric,
      n_estimators=1000,
      learning_rate=0.02,
      max_features=0.5,
      query_subsample=0.5,
      max_leaf_nodes=10,
      min_samples_leaf=64,
      verbose=1,
  )

  model.fit(x_train.todense(), y_train, qid_train)
  preds = model.predict(x_test)
  print('Random ranking:', metric.calc_mean_random(qid_test, y_test))
  print('Our model:', metric.calc_mean(qid_test, y_test, preds))
  import ipdb; ipdb.set_trace()


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
