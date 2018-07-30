import hashlib
import os
import warnings

import pydash as _
from tabulate import tabulate

from logger import Logger
from utils import append_create

class ExperimentContext(object):
  def __init__(self):
    pass

  def __enter__(self):
    pass

  def __exit__(self, *args):
    pass

class Experiment(object):
  def __init__(self, params):
    self.dirty_worktree = False
    if os.popen('git status --untracked-files=no --porcelain').read() != '':
      self.dirty_worktree = True
      warnings.warn('git tree dirty! git hash will not correspond to the codebase!')
    self.log = Logger()
    self.name = None
    self.is_train = None
    self.is_test = None
    self.epoch_num = None
    self.params = params
    self.log.table(_.map_values(self.params, lambda val: [str(val)]),
                   'params')
    self.metrics = {}
    with open('params_' + self.model_name, 'w+') as f:
      obj = _.map_values(self.params, lambda val: [str(val)])
      keys = sorted(list(obj.keys()))
      vals = list(zip(*[obj[key] for key in keys]))
      f.write(tabulate(vals, headers=keys, tablefmt='orgtbl'))

  @property
  def model_name(self):
    param_names = sorted([key for key in self.params.keys() if key != 'load_model'])
    param_strings = [name + '=' + str(self.params[name]) for name in param_names]
    return 'model_' + hashlib.sha256(str.encode('_'.join(param_strings))).hexdigest()

  def log_multiple_metrics(self, metrics, step=None):
    metric_names = sorted(list(metrics.keys()))
    self.log.report(*[metrics[name] for name in metric_names])

  def log_metric(self, name, val):
    append_create(self.metrics, name, val)
    self.log.report(name, val)

  def set_name(self, name):
    self.name = name

  def train(self):
    self.is_train = True
    self.is_test = False
    return ExperimentContext()

  def test(self):
    self.is_train = False
    self.is_test = True
    return ExperimentContext()

  def log_current_epoch(self, epoch_num):
    self.epoch_num = epoch_num
    self.log.status('epoch ' + str(epoch_num))

  def log_epoch_end(self, epoch_num):
    pass
