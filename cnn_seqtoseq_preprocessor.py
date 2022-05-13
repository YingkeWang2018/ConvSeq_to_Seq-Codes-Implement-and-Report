import os
import spacy
from torchtext.legacy.data import Field, BucketIterator, TabularDataset


def preprocess_data(src, trg):
  os.system('python -m spacy download fr')
  os.system('python -m spacy download en')

  spacy_fr = spacy.load('fr')
  spacy_en = spacy.load('en')


  fields = {'English': ('src', src), 'French': ('trg', trg)}
  train_data, valid_data = TabularDataset.splits(path='', train='train_small.csv', validation='val_small.csv',
                                                  format='csv', fields=fields, skip_header=False)
  test_data = TabularDataset(path='test_small.csv', format='csv', fields=fields, skip_header=False)
  src.build_vocab(train_data, min_freq=2)
  trg.build_vocab(train_data, min_freq=2)
  return train_data, valid_data, test_data, src, trg