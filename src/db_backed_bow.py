from cached_bow import CachedBoW

class DBBoW(CachedBoW):
  def __init__(self, docs_name, cursor, query_template, token_idx_lookup=None, unk_idx=1):
    super().__init__(docs_name, token_idx_lookup=token_idx_lookup, unk_idx=unk_idx)
    self.cursor = cursor
    self.query_template = query_template

  def _get_doc_text(self, doc_id):
    self.cursor.execute(self.query_template.format(doc_id))
    return self.cursor.fetchone()['text']
