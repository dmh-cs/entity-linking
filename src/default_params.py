from pyrsistent import m

default_paths = m(lookups='../entity-linking-preprocessing/lookups.pkl',
                  page_id_order='../entity-linking-preprocessing/page_id_order.pkl')
default_train_params = m(batch_size=100,
                         debug=False,
                         num_epochs=1,
                         train_size=0.8,
                         dropout_drop_prob=0.4)
default_model_params = m(use_adaptive_softmax=False,
                         use_ranking_loss=False,
                         use_hardcoded_cutoffs=True,
                         embed_len=100,
                         word_embed_len=100,
                         num_candidates=30,
                         word_embedding_set='glove',
                         local_encoder_lstm_size=100,
                         document_encoder_lstm_size=100,
                         num_lstm_layers=2,
                         ablation=['prior', 'local_context', 'document_context'])
default_run_params = m(load_model=False,
                       comments='')
