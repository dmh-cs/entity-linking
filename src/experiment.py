class Experiment(object):
  def __init__(self):
    self.log = lambda *args: print('|'.join([str(arg) for arg in args]))
    self.name = None
    self.is_train = None
    self.is_test = None
    self.epoch_num = None

  def log_multiple_metrics(self, metrics):
    for name, val in metrics.items():
      self.log_metric(name, val)

  def log_metric(self, name, val):
    self.log(name, val)

  def set_name(self, name):
    self.name = name

  def log_multiple_params(self, params):
    for name, val in params.items():
      self.log_param(name, val)

  def log_param(self, name, val):
    self.log(name, val)

  def train(self):
    self.is_train = True
    self.is_test = False

  def test(self):
    self.is_train = False
    self.is_test = True

  def log_current_epoch(self, epoch_num):
    self.epoch_num = epoch_num
    self.log(epoch_num)

  def log_epoch_end(self):
    pass
