from tabulate import tabulate

class Logger(object):
  def __init__(self):
    self.status = print
    self.report = lambda *args: print('|'.join([str(arg) for arg in args]))

  def table(self, obj, name=''):
    print('-----' + name + '-----')
    keys = sorted(list(obj.keys()))
    vals = list(zip(*[obj[key] for key in keys]))
    print(tabulate(vals, headers=keys, tablefmt='orgtbl'))
    print('----------------')
