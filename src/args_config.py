import pydash as _

from rabbit_ml import list_arg, optional_arg

train_args = [
  {'name': 'batch_size','type': int,                      'default': 512},
  {'name': 'dropout_keep_prob','type': float,             'default': 0.5},
  {'name': 'learning_rate','type': float,                 'default': 1e-3},
  {'name': 'margin','type': float,                        'default': 0.1},
  {'name': 'max_num_epochs','type': int,                  'default': 10},
  {'name': 'num_pages_to_use', 'type': int,               'default': 10000},
  {'name': 'stop_by','type': str,                         'default': 'acc'},
  {'name': 'train_on_conll','type': 'flag',               'default': False},
  {'name': 'train_size','type': float,                    'default': 1.0},
  {'name': 'use_hinge','type': 'flag',                    'default': False},
  {'name': 'use_pairwise','type': 'flag',                 'default': False},
  {'name':'page_id_order_path','type': str, 'default': '../wp-preprocessing-el/page_id_order.pkl_local'},
  {'name':'stop_after_n_bad_epochs','type': int,          'default': 2},
  {'name':'use_sequential_sampler','type': 'flag',        'default': False},
  {'name':'xgboost_test_path','type': optional_arg(str),  'default': None},
  {'name':'xgboost_train_path','type': optional_arg(str), 'default': None},
]

run_args = [
  {'name': 'batch_size','type': int,                      'default': 512},
  {'name': 'env_path','type': str,                        'default': '.env'},
  {'name': 'idf_path','type': str,                        'default': './wiki_idf_stem.json'},
  {'name': 'lookups_path','type': str, 'default': '../wp-preprocessing-el/lookups.pkl_local'},
  {'name': 'lookups_path','type': str, 'default': '../wp-preprocessing-el/lookups.pkl_local'},
  {'name': 'txt_dataset_path', 'type': optional_arg(str), 'default': None},
  {'name': 'val_txt_dataset_path', 'type': optional_arg(str), 'default': None},
  {'name': 'pkl_dataset_prefix', 'type': optional_arg(str), 'default': None},
  {'name': 'use_custom','type': 'flag',                   'default': False},
  {'name':'xgboost_path','type': optional_arg(str),       'default': None},
]

model_args =  [
  {'name': 'just_str_sim','type': 'flag',        'default': False},
  {'name': 'just_tfidf','type': 'flag',          'default': False},
  {'name': 'just_wiki2vec','type': 'flag',       'default': False},
  {'name': 'hidden_sizes','type': list_arg(int), 'default': [100, 50]},
]

args = [_.assign({'for': 'train_params'}, arg) for arg in train_args] + [_.assign({'for': 'run_params'}, arg) for arg in run_args] + [_.assign({'for': 'model_params'}, arg) for arg in model_args]
