from cached_bow import CachedBoW

class DBBoW(CachedBoW):
  def __init__(self, docs_name, cursor, query_template):
    super().__init__(docs_name)
    self.cursor = cursor
    self.query_template = query_template

  def _get_doc_text(self, doc_id):
    self.cursor.execute(self.query_template.format(doc_id))
    result = self.cursor.fetchone()
    if result is None: raise IndexError
    return result['text']
