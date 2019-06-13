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
      row_idxs = [self.doc_id_to_row.get(i) for i in idx]
      without_none = [self.doc_id_to_row[i]
                      for i in idx
                      if i in self.doc_id_to_row]
      to_result_num = [result_num
                       for result_num, index in enumerate(idx)
                       if index in self.doc_id_to_row]
      if (len(row_idxs) != len(without_none)) and not self.use_default:
        raise IndexError
      rows = self.mat[without_none]
      result = [{} if row_idx is not None else self.default_value
                for row_idx in row_idxs]
      row_nums, token_idxs = rows.nonzero()
      cnts = rows[row_nums, token_idxs].tolist()[0]
      for row_num, token_idx, cnt in zip(row_nums, token_idxs, cnts):
        if self.token_idx_mapping is not None:
          result[to_result_num[row_num]][self.token_idx_mapping[token_idx]] = cnt
        else:
          result[to_result_num[row_num]][token_idx] = cnt
      return result
    else:
      return pipe(idx, self._get_sparse_row, self._to_lookup)
