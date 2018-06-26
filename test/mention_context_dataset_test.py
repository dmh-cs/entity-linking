from mention_context_dataset import MentionContextDataset

def get_mock_cursor():
  return None

def test_mention_context_dataset():
  cursor = get_mock_cursor()
  page_id_order = [3, 1, 2]
  batch_size = 5
  num_mentions = 5
  entity_candidates_lookup = {1: [1],
                              0: [0, 1],
                              3: [0, 1],
                              4: [0, 1],
                              2: [2]}
  dataset = MentionContextDataset(cursor,
                                  page_id_order,
                                  entity_candidates_lookup,
                                  batch_size,
                                  num_mentions)
  dataset._mention_infos = {0: {'mention': 'bb', 'offset': 9, 'page_id': 2, 'entity_id': 0, 'mention_id': 0},
                            1: {'mention': 'aa', 'offset': 6, 'page_id': 2, 'entity_id': 1, 'mention_id': 1},
                            2: {'mention': 'cc', 'offset': 0, 'page_id': 1, 'entity_id': 2, 'mention_id': 2},
                            3: {'mention': 'bb', 'offset': 3, 'page_id': 1, 'entity_id': 0, 'mention_id': 3},
                            4: {'mention': 'bb', 'offset': 3, 'page_id': 0, 'entity_id': 1, 'mention_id': 4}}
  dataset._page_id_sentences_lookup = {2: ['a b c', 'aa bb'],
                                       1: ['cc bb', 'c b a'],
                                       0: ['dd bb', 'a b c']}
  dataset._document_mention_lookup = {2: [0, 1],
                                      1: [1, 2],
                                      0: [1]}
  expected_data = [{'sentence_splits': [['aa', 'bb'], ['bb']],
                    'label': 0,
                    'document_mention_indices': [1],
                    'candidates': [0, 1]},
                   {'sentence_splits': [['aa'], ['aa', 'bb']],
                    'label': 1,
                    'document_mention_indices': [1, 2],
                    'candidates': [1]},
                   {'sentence_splits': [['cc'], ['cc', 'bb']],
                    'label': 2,
                    'document_mention_indices': [0, 1],
                    'candidates': [2]},
                   {'sentence_splits': [['cc', 'bb'], ['bb']],
                    'label': 0,
                    'document_mention_indices': [1],
                    'candidates': [0, 1]},
                   {'sentence_splits': [['dd', 'bb'], ['bb']],
                    'label': 1,
                    'document_mention_indices': [1, 2],
                    'candidates': [0, 1]}]
  dataset_values = list(dataset)
  assert expected_data == dataset_values
