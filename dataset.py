from collections import Counter
from itertools import chain
import random
from collections import defaultdict
import numpy as np
import torch

def collate_fn(batch):
    
    PAD = Vocabulary.PAD_TOKEN_IDX
    
    src = [sample[0] for sample in batch]
    tgt = [sample[1] for sample in batch]
    src_lengths = [len(sample) for sample in src]
    tgt_lengths = [len(sample) for sample in tgt]
    max_src_len = max(src_lengths)
    max_tgt_len = max(tgt_lengths)
    for i in range(len(batch)):
        src[i] = np.asarray(src[i], dtype=np.int64)
        src[i] = np.pad(src[i], (0, max_src_len-len(src[i])), mode="constant", constant_values=PAD)
        tgt[i] = np.asarray(tgt[i], dtype=np.int64)
        tgt[i] = np.pad(tgt[i], (0, max_tgt_len-len(tgt[i])), mode="constant", constant_values=PAD)

    src_sentences = torch.LongTensor(src)
    trg_sentences = torch.LongTensor(tgt)

    _, sorted_indices = torch.sort(torch.LongTensor(src_lengths), dim=0, descending=True)
    src_sentences = src_sentences.index_select(0, sorted_indices)
    trg_sentences = trg_sentences.index_select(0, sorted_indices)

    return src_sentences, trg_sentences

class Vocabulary():
    PAD_TOKEN = '<PAD>'
    PAD_TOKEN_IDX = 0
    UNK_TOKEN = '<UNK>'
    UNK_TOKEN_IDX = 1
    SOS_TOKEN = '<SOS>'
    SOS_TOKEN_IDX = 2
    EOS_TOKEN = '<EOS>'
    EOS_TOKEN_IDX = 3

    def __init__(self, path):
        with open(path, mode='r', encoding='utf-8') as f:
            self._sentences = [line.split() for line in f]

    def build_vocab(self):
        SPECIAL_TOKENS = [Vocabulary.PAD_TOKEN, Vocabulary.UNK_TOKEN, Vocabulary.SOS_TOKEN, Vocabulary.EOS_TOKEN]
        self.idx2word = SPECIAL_TOKENS + [word for word, _ in Counter(chain(*self._sentences)).items()]
        self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}
    
    def set_vocab(self, word2idx, idx2word):
        self.word2idx = word2idx
        self.idx2word = idx2word
    
    def __getitem__(self, index):
        return self._sentences[index]
    
    def __len__(self):
        return len(self._sentences)

class Data_loader():
    def __init__(self, src, trg):
        self._src = src
        self._trg = trg
        
    def __getitem__(self, index):

        raw_src_sentence=self._src[index]
        raw_trg_sentence=self._trg[index]
        src_word2idx=self._src.word2idx
        trg_word2idx=self._trg.word2idx
        src_sentence = [src_word2idx[word] if word in src_word2idx.keys() else Vocabulary.UNK_TOKEN_IDX for word in raw_src_sentence]
        trg_sentence = [trg_word2idx[word] if word in trg_word2idx.keys() else Vocabulary.UNK_TOKEN_IDX for word in raw_trg_sentence]

        trg_sentence.insert(0, Vocabulary.SOS_TOKEN_IDX)
        trg_sentence.append(Vocabulary.EOS_TOKEN_IDX)

        return src_sentence, trg_sentence

    def __len__(self):
        return len(self._src)