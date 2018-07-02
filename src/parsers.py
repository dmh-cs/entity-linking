# coding: utf-8

import re
import pydash as _
from nltk.data              import load
from nltk.tokenize.treebank import TreebankWordTokenizer

_treebank_word_tokenizer = TreebankWordTokenizer()

improved_open_quote_regex = re.compile(u'([«“‘„]|[`]+)', re.U)
improved_close_quote_regex = re.compile(u'([»”’])', re.U)
other_improved_open_quote_regex = re.compile(u'(\'\')', re.U)
other_improved_close_quote_regex = re.compile(u'(\'\')', re.U)
improved_punct_regex = re.compile(r'([^\.])(\.)([\]\)}>"\'' u'»”’ ' r']*)\s*$', re.U)
_treebank_word_tokenizer.STARTING_QUOTES.insert(0, (improved_open_quote_regex, r' \1 '))
_treebank_word_tokenizer.ENDING_QUOTES.insert(0, (improved_close_quote_regex, r' \1 '))
_treebank_word_tokenizer.STARTING_QUOTES.insert(0, (other_improved_open_quote_regex, r' \1 '))
_treebank_word_tokenizer.ENDING_QUOTES.insert(0, (other_improved_close_quote_regex, r' \1 '))
_treebank_word_tokenizer.PUNCTUATION.insert(0, (improved_punct_regex, r'\1 \2 \3 '))

def parse_for_sentence_spans(page_content):
  tokenizer = load('tokenizers/punkt/{0}.pickle'.format('english'))
  return list(tokenizer.span_tokenize(page_content))

def parse_for_sentences(page_content):
  tokenizer = load('tokenizers/punkt/{0}.pickle'.format('english'))
  return list(tokenizer.tokenize(page_content))

def parse_for_tokens(sentence):
  return list(_treebank_word_tokenizer.tokenize(sentence))

def parse_text_for_tokens(text):
  sentences = parse_for_sentences(text)
  return _.flatten([parse_for_tokens(sentence) for sentence in sentences])
