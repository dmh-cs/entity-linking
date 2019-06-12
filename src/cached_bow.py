import ast
from pymongo import MongoClient
from abc import ABC, abstractmethod
from collections import Counter
import pickle
from pathlib import Path

from parsers import parse_text_for_tokens

class CachedBoW(ABC):
  def __init__(self, docs_name):
    self.docs_name = docs_name
    self.cache_path = Path('./cache/').joinpath(docs_name)
    self.cache_path.mkdir(parents=True, exist_ok=True) # pylint: disable=no-member
    # pylint bug https://github.com/PyCQA/pylint/issues/1660
    self.mongo_client = MongoClient()
    self.mongo_db = self.mongo_client['{}_cache'.format(self.docs_name)]

  def _get_doc_name(self, doc_id): return '{}_{}'.format(self.docs_name, doc_id)

  def _get_cache_path(self, doc_id): return self.cache_path.joinpath(self._get_doc_name(doc_id))

  def _cached_get(self, doc_id):
    if hasattr(doc_id, '__iter__'):
      query = {'_id': {'$in': [str(d_id) for d_id in doc_id]}}
      result = {int(result['_id']): ast.literal_eval(result['content'])
                for result in self.mongo_db['{}_cache'.format(self.docs_name)].find(query)}
      return [result.get(d_id, None) for d_id in doc_id]
    else:
      query = {'_id': str(doc_id)}
      result = self.mongo_db['{}_cache'.format(self.docs_name)].find_one(query)
      return ast.literal_eval(result['content'])

  def _cache_result(self, doc_id, bow):
    if hasattr(doc_id, '__iter__'): raise NotImplementedError
    with open(self._get_cache_path(doc_id), 'wb') as fh:
      pickle.dump(bow, fh)

  def _get_bow(self, doc_id):
    text = self._get_doc_text(doc_id)
    tokenized = parse_text_for_tokens(text)
    counts = dict(Counter(tokenized))
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
