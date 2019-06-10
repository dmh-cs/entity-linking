from pathlib import Path

from cached_bow import CachedBoW

class FileCachedBoW(CachedBoW):
  def __init__(self, docs_name, docs_path, token_idx_lookup=None, unk_idx=1):
    super().__init__(docs_name, token_idx_lookup=token_idx_lookup, unk_idx=1)
    self.docs_path = Path(docs_path)

  def _get_doc_path(self, doc_id): return self.docs_path.joinpath(self._get_doc_name(doc_id))

  def _get_doc_text(self, doc_id):
    with open(self._get_doc_path(doc_id)) as fh: return fh.read()
