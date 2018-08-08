from typing import List
import hashlib
import os
import warnings

import pydash as _
from git import Repo
import chalk
from toolz import compose

from logger import Logger

warnings.showwarning = lambda *args, **kwargs: print(chalk.yellow(args[0]))

class ExperimentContext(object):
  def __init__(self, separator: str, train_or_test: bool, run_name: str, fields: List[str]):
    self.file_handle = open('./results_' + train_or_test + '_' + run_name, 'a+')
    self.fields = fields
    self.separator = separator

  def __enter__(self):
    self.file_handle.write(self.separator.join(sorted(self.fields) + ['batch', 'epoch']) + '\n')
    return self.file_handle

  def __exit__(self, *args):
    self.file_handle.close()

class Experiment(object):
  def __init__(self, params):
    self.separator = '|'
    self.training = None
    self.log = Logger()
    self.epoch_num = None
    self.params = params
    self.metrics = {}
    self.file_handle = None
    self._repo = Repo('./')
    self.dirty_worktree = False
    if os.popen('git status --untracked-files=no --porcelain').read() != '':
      self.dirty_worktree = True
      warnings.warn('git tree dirty! git hash will not correspond to the codebase!')

  @property
  def train_or_test(self):
    assert self.training is not None
    if self.training:
      return 'train'
    else:
      return 'test'

  @property
  def ablation_string(self):
    return '_'.join(self.params['ablation'])

  @property
  def model_name(self):
    param_names = sorted([key for key in self.params.keys() if key not in ['ablation', 'load_model']])
    param_strings = [name + '=' + str(self.params[name]) for name in param_names]
    hash_string = hashlib.sha256(str.encode('_'.join(param_strings))).hexdigest()
    return 'model_' + hash_string

  @property
  def run_name(self):
    return self.model_name + '_' + self.ablation_string

  def record_metrics(self, metrics, batch_num=None):
    metric_names = sorted(list(metrics.keys()))
    vals = [str(metrics[name]) for name in metric_names] + [str(batch_num), str(self.epoch_num)]
    self.file_handle.write(self.separator.join(vals) + '\n')

  def update_epoch(self, epoch_num):
    self.epoch_num = epoch_num

  def train(self, fields):
    self.training = True
    self._write_details()
    context = ExperimentContext(self.separator, self.train_or_test, self.run_name, fields)
    self.file_handle = context.file_handle
    return context

  def test(self, fields):
    self.training = False
    self._write_details()
    context = ExperimentContext(self.separator, self.train_or_test, self.run_name, fields)
    self.file_handle = context.file_handle
    return context

  def _write_details(self):
    master = self._repo.head.reference
    with open('params_' + self.run_name, 'w+') as f:
      for name, val in self.params.items():
        f.write(name + self.separator + str(val) + '\n')
      f.write('commit hash' + self.separator + str(master.commit.hexsha) + '\n')
      f.write('commit msg' + self.separator + str(master.commit.message))
      f.write('dirty worktree?' + self.separator + str(self.dirty_worktree) + '\n')
