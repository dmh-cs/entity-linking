from abc import ABC, abstractmethod
from collections import Counter
import pickle
from pathlib import Path

from parsers import parse_text_for_tokens

class CachedBoW(ABC):
  def __init__(self, docs_name, token_idx_lookup=None, unk_idx=1):
    self.token_idx_lookup = token_idx_lookup
    self.docs_name = docs_name
    self.cache_path = Path('./cache/').joinpath(docs_name)
    self.unk_idx = unk_idx
    self.cache_path.mkdir(parents=True, exist_ok=True) # pylint: disable=no-member
                                                       # pylint bug https://github.com/PyCQA/pylint/issues/1660

  def _get_doc_name(self, doc_id): return '{}_{}'.format(self.docs_name, doc_id)

  def _get_cache_path(self, doc_id): return self.cache_path.joinpath(self._get_doc_name(doc_id))

  def _cached_get(self, doc_id):
    with open(self._get_cache_path(doc_id), 'rb') as fh:
      return pickle.load(fh)

  def _cache_result(self, doc_id, bow):
    with open(self._get_cache_path(doc_id), 'wb') as fh:
      pickle.dump(bow, fh)

  def _get_bow(self, doc_id):
    text = self._get_doc_text(doc_id)
    tokenized = parse_text_for_tokens(text)
    if self.token_idx_lookup is None:
      counts = dict(Counter(tokenized))
    else:
      counts = dict(Counter(self.token_idx_lookup.get(token, self.unk_idx)
                            for token in tokenized))
    return counts

  def __getitem__(self, doc_id):
    try:
      bow = self._cached_get(doc_id)
    except FileNotFoundError:
      bow = self._get_bow(doc_id)
      self._cache_result(doc_id, bow)
    return bow

  @abstractmethod
  def _get_doc_text(self, doc_id): pass
