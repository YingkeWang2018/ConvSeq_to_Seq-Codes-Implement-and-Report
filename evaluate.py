from dataset import Vocabulary, Data_loader, collate_fn
from model import BiLSTM
import torch
from tqdm import tqdm,trange
from config import *
import random
from utlis import prepare_dataset,BLEU_score
import numpy as np

device = DEVICE_DEFAULT
embedding_dim = EMBEDDING_SIZE_DEFAULT
hidden_dim = HIDDEN_SIZE_DEFAULT

def Evaluate():
    
    english_train, french_train = prepare_dataset(src_path='./data/train_en_small.txt',trg_path='./data/train_fr_small.txt',build=True)
    model = BiLSTM(english_train, french_train, embedding_size=embedding_dim, hidden_size=hidden_dim).to(device)
    model.load_state_dict(torch.load("Bi_LSTM.pth", map_location=device))
    english_test,french_test = prepare_dataset(src_path='./data/test_en_small.txt',trg_path='./data/test_fr_small.txt',build=False)
    french_test.set_vocab(french_train.word2idx, french_train.idx2word)
    english_test.set_vocab(english_train.word2idx, english_train.idx2word)
    dataset = Data_loader(src=english_test, trg=french_test)
    
    f = open('prediction.txt','w', encoding='utf-8')

    for english, french in tqdm(dataset):
        translated, attention = model.translate(torch.Tensor(english).to(dtype=torch.long, device=device))
        source_text = [english_train.idx2word[idx] for idx in english]
        target_text = [french_train.idx2word[idx] for idx in french if idx != Vocabulary.SOS_TOKEN_IDX and idx != Vocabulary.EOS_TOKEN_IDX]
        translated_text = [french_train.idx2word[idx] for idx in translated if idx != Vocabulary.EOS_TOKEN_IDX]
        f.write(' '.join(translated_text) + '\n')
    f.close()

if __name__ == "__main__":
    random.seed(RANDOM_SEED_DEFAULT)
    torch.manual_seed(RANDOM_SEED_DEFAULT)
    Evaluate()

    cand=[]
    f1=open("prediction.txt","r",encoding="utf8")
    for i in f1:
        cand.append(i.strip())
    ref=[]
    f2=open("./data/test_fr_small.txt","r",encoding="utf8")
    for j in f2:
        ref.append(j.strip())
    score=[]
    for i in trange(len(cand)):
        ret=BLEU_score(cand[i], ref[i], n=4)
        score.append(ret)
    
    print(np.mean(score))
  
