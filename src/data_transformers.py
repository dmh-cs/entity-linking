def transform_raw_datasets(raw_datasets):
  tokens = parse_datasets_into_tokens(raw_datasets)
  alphabet = _build_alphabet(tokens)
  embeddings = _build_embeddings(alphabet, tokens)
  return _build_datasets(raw_datasets)
