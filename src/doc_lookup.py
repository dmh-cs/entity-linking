from toolz import pipe

from scipy.sparse import load_npz

class DocLookup():
  def __init__(self,
               sparse_mat_path,
               doc_id_to_row,
               token_idx_mapping=None,
               default_value=None,
               use_default=False):
    self.sparse_mat_path = sparse_mat_path
    self.mat = load_npz(sparse_mat_path)
    self.default_value = default_value
    self.use_default = use_default
    self.doc_id_to_row = doc_id_to_row
    self.token_idx_mapping = token_idx_mapping

  def _to_lookup(self, sparse_row):
    if sparse_row is None:
      return self.default_value
    token_idxs = sparse_row.nonzero()[1]
    cnts = sparse_row[0, token_idxs].toarray().reshape(-1).tolist()
    if self.token_idx_mapping is not None:
      return dict(zip((self.token_idx_mapping[idx] for idx in token_idxs),
                      cnts))
    else:
      return dict(zip(token_idxs, cnts))

  def _get_sparse_row(self, idx):
    if self.use_default:
      try:
        row_idx = self.doc_id_to_row[idx]
      except KeyError:
        return None
    else:
      row_idx = self.doc_id_to_row[idx]
    return self.mat[row_idx]

  def __getitem__(self, idx):
    if hasattr(idx, '__iter__'):
      return [pipe(index, self._get_sparse_row, self._to_lookup)
              for index in idx]
    else:
      return pipe(idx, self._get_sparse_row, self._to_lookup)
