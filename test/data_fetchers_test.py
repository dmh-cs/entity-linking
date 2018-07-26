import data_fetchers as df

def test_get_random_indexes():
  result = df.get_random_indexes(10, [2, 8], 10)
  assert 2 not in result
  assert 8 not in result
  assert len(result) == 10

def test_get_random_indexes_too_long():
  caught = False
  try:
    df.get_random_indexes(10, [2, 8], 12)
  except Exception as e:
    assert isinstance(e, ValueError)
    caught = True
  assert caught
