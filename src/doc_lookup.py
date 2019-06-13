from scipy.sparse import load_npz

class DocLookup():
  def __init__(self,
               sparse_mat_path,
               doc_id_to_row,
               token_idx_to_str=None,
               default_value=None,
               use_default=False):
    self.sparse_mat_path = sparse_mat_path
    self.mat = load_npz(sparse_mat_path)
    self.default_value = default_value
    self.use_default = use_default
    self.doc_id_to_row = doc_id_to_row
    self.token_idx_to_str = token_idx_to_str

  def _to_lookup(self, sparse_row):
    token_idxs = sparse_row.nonzero()[1]
    cnts = sparse_row[0, token_idxs].toarray().reshape(-1).tolist()
    if self.token_idx_to_str is not None:
      return dict(zip((self.token_idx_to_str[idx] for idx in token_idxs),
                      cnts))
    else:
      return dict(zip(token_idxs, cnts))

  def _get_sparse_rows(self, idx):
    if hasattr(idx, '__iter__'):
      return [self[index] for index in idx]
    if self.use_default:
      try:               return self[idx]
      except IndexError: return self.default_value

  def __getitem__(self, idx):
    if hasattr(idx, '__iter__'):
      sparse_rows =  self._get_sparse_rows([self.doc_id_to_row[index]
                                            for index in idx])
      return [self._to_lookup(sparse_row) for sparse_row in sparse_rows]
    else:
      sparse_row = self._get_sparse_rows(self.doc_id_to_row[idx])
      return self._to_lookup(sparse_row)
