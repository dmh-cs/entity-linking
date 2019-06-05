from rabbit_ml import list_arg, optional_arg

args =  [{'name': 'batch_size',       'for': 'train_params', 'type': int, 'default': 512},
         {'name': 'batch_size',       'for': 'run_params', 'type': int, 'default': 512},
         {'name': 'env_path',         'for': 'run_params', 'type': str, 'default': '.env'},
         {'name': 'hidden_sizes',     'for': 'model_params', 'type': list_arg(str), 'default': [100]},
         {'name': 'idf_path',         'for': 'run_params', 'type': str, 'default': './wiki_idf_stem.json'},
         {'name': 'just_tfidf',       'for': 'model_params', 'type': 'flag', 'default': False},
         {'name': 'just_str_sim',       'for': 'model_params', 'type': 'flag', 'default': False},
         {'name': 'lookups_path',     'for': 'run_params', 'type': str, 'default': '../wp-preprocessing-el/lookups.pkl_local'},
         {'name': 'num_pages_to_use', 'for': 'train_params', 'type': int, 'default': 10000},
         {'name': 'page_id_order_path', 'for': 'train_params', 'type': str, 'default': '../wp-preprocessing-el/page_id_order.pkl_local'},
         {'name': 'train_size',       'for': 'train_params', 'type': float, 'default': 1.0},
         {'name': 'use_custom',        'for': 'run_params', 'type': 'flag', 'default': False},
         {'name': 'use_hinge',        'for': 'train_params', 'type': 'flag', 'default': False},
         {'name': 'use_pairwise',     'for': 'train_params', 'type': 'flag', 'default': False},
         {'name': 'num_epochs',       'for': 'train_params', 'type': int, 'default': 5}]
