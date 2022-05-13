from dataset import Data_loader, collate_fn
from model import BiLSTM
import torch

import torch.utils
import random
from tqdm import tqdm, trange
from config import *
from utlis import prepare_dataset
import os
import dill

device = DEVICE_DEFAULT
embedding_dim = EMBEDDING_SIZE_DEFAULT
hidden_dim = HIDDEN_SIZE_DEFAULT
max_epoch = MAX_EPOCH_DEFAULT
batch_size = BATCH_SIZE_DEFAULT
teacher_force=TEACHER_FORCE_DEFAULT

def train():
    
    english, french = prepare_dataset(src_path='./data/train_en_small.txt',trg_path='./data/train_fr_small.txt',build=True)
    dataset = Data_loader(src=english, trg=french)

    model = BiLSTM(english, french, embedding_size=embedding_dim, hidden_size=hidden_dim).to(device)
    if os.path.isfile("pre_trained.pth"):
        model.load_state_dict(torch.load("pre_trained.pth", map_location=device))

    optimizer = torch.optim.Adam(model.parameters())
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=2, batch_size= batch_size, shuffle=True, batch_sampler=None, collate_fn=collate_fn)

    train_losses=[]
    for epoch in range(max_epoch):
        idx=0
        for src_sentence, trg_sentence in tqdm(dataloader):
            idx+=1
            optimizer.zero_grad()
            src_sentence, trg_sentence = src_sentence.to(device), trg_sentence.to(device)
            loss = model(src_sentence, trg_sentence, teacher_force=teacher_force)
            loss.backward()
            optimizer.step()
            if idx%500 ==0:
                print(f"Current {epoch}Epoch, {idx} record, the loss is {loss.item()}")
                torch.save(model.state_dict(), "Bi_LSTM.pth")
        train_losses.append(loss.item())
    
    traiable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"There are {traiable_parameters} trainable parameters")

    with open("train_losses", "wb") as dill_file:
        dill.dump(train_losses, dill_file)

if __name__ == "__main__":
    random.seed(RANDOM_SEED_DEFAULT)
    torch.manual_seed(RANDOM_SEED_DEFAULT)
    train()
  