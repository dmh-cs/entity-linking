from pyrsistent import m

default_paths = m(lookups='../entity-linking-preprocessing/lookups.pkl',
                  page_id_order='../entity-linking-preprocessing/page_id_order.pkl',
                  env='.env')
default_train_params = m(batch_size=100,
                         dataset_limit=None,
                         debug=False,
                         num_epochs=1,
                         min_mentions=1,
                         train_size=0.8,
                         margin=0.1,
                         dropout_drop_prob=0.4,
                         start_from_page_num=0,
                         clip_grad=0.01)
default_model_params = m(num_cnn_local_filters=50,
                         embed_len=100,
                         word_embed_len=100,
                         num_candidates=30,
                         word_embedding_set='glove',
                         local_encoder_lstm_size=100,
                         document_encoder_lstm_size=100,
                         num_lstm_layers=2,
                         ablation=['prior', 'local_context', 'document_context'])
default_run_params = m(load_model=False,
                       cheat=False,
                       comments='',
                       buffer_scale=1)
