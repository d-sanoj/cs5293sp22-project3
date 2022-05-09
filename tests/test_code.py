import sys
sys.path.append('..')
import unredactor
import pytest
import nltk
import os.path

def test_readinput():
    if os.path.exists('../unredactor.tsv'):
        assert True

def test_removewhile():
    data = ' Mike'
    final = unredactor.remove_whitespace(data)
    assert final == 'Mike'

def test_stopwords():
    data = 'is'
    final = unredactor.remove_stopwords(data)
    assert final == []

def test_punc():
    data = "'"
    final = unredactor.remove_punct(data)
    assert final == []
