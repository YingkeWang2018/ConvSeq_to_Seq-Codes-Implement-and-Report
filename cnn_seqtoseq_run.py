import torch
import random
import time
import numpy as np
import spacy
from torchtext.legacy.data import Field, BucketIterator, TabularDataset
from torchtext.data.metrics import bleu_score
import cnn_seqtoseq_config as config
from cnn_seqtoseq_helper import multi_acc, count_parameters, epoch_time, translate_text
from cnn_seqtoseq_model import Encoder, Decoder, CnnSeqToSeq
from cnn_seqtoseq_preprocessor import preprocess_data
import torch.nn as nn
import torch.optim as optim
import os
import dill



def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    accs = []
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        loss = criterion(output, trg)
        acc = multi_acc(output, trg)
        accs.append(acc)
        loss.backward()
        # add clipping for normalization, or loss is too large
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator), float(sum(accs) / len(iterator))

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    accs = []

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg[:, :-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            acc = multi_acc(output, trg)
            accs.append(acc)
            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator), float(sum(accs) / len(iterator))


def tokenize_fr(text):
    """
    Tokenizes French text from a string into a list of strings
    """
    return [tok.text for tok in spacy_fr.tokenizer(text)]


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

if __name__ == '__main__':
    SEED = 1234

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    os.system('python -m spacy download fr')
    os.system('python -m spacy download en')

    spacy_fr = spacy.load('fr')
    spacy_en = spacy.load('en')
    SRC = Field(tokenize=tokenize_en,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True,
                batch_first=True)

    TRG = Field(tokenize=tokenize_fr,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True,
                batch_first=True)
    train_data, valid_data, test_data, SRC, TRG = preprocess_data(SRC, TRG)

    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    TRG_PADDING_IDX = TRG.vocab.stoi[TRG.pad_token]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=config.BATCH_SIZE,
        device=device,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src)
    )

    encoder = Encoder(INPUT_DIM, config.EMB_DIM, config.HID_DIM, config.ENCODER_LAYERS, config.ENCODER_KERNEL_SIZE,
                      config.ENCODER_DROPOUT, device)
    decoder = Decoder(OUTPUT_DIM, config.EMB_DIM, config.HID_DIM, config.DECODER_LAYERS, config.DECODER_KERNEL_SIZE,
                      config.DECODER_DROPOUT, TRG_PADDING_IDX, device)

    model = CnnSeqToSeq(encoder, decoder).to(device)

    print(f'The model has {count_parameters(model)} parameters')
    #optimaizer
    optimizer = optim.Adam(model.parameters())
    loss_func = nn.CrossEntropyLoss(ignore_index=TRG_PADDING_IDX)

    train_losses = []
    validation_losses = []
    best_valid_loss = (float('inf'), 0)

    for epoch in range(config.N_EPOCHS):
        start_time = time.time()
        train_loss = train(model, train_iterator, optimizer, loss_func, config.CLIP)
        valid_loss = evaluate(model, valid_iterator, loss_func)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        valid_loss_a = valid_loss[0]
        best_loss_a = best_valid_loss[0]

        if valid_loss_a < best_loss_a:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'model-EMB_DIM = 100 HID_DIM = 512 ENC_LAYERS = 10 DEC_LAYERS = 10.pt')

        print(f'Epoch: {epoch + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'Train Loss: {round(train_loss[0], 2)}')
        print(f'Train Accuracy: {round(train_loss[1], 2)}%')
        print(f'Validation Loss: {round(valid_loss[0], 2)}')
        print(f'Validation Accuracy: {round(valid_loss[1], 2)}%')
        train_losses.append(train_loss[0])
        validation_losses.append(valid_loss[0])

    with open("training_loss, EMB_DIM = 256 HID_DIM = 256 ENC_LAYERS = 10 DEC_LAYERS = 10", "wb") as dill_file:
        dill.dump(train_losses, dill_file)

    with open("validation_loss, EMB_DIM = 256 HID_DIM = 256 ENC_LAYERS = 10 DEC_LAYERS = 10", "wb") as dill_file:
        dill.dump(validation_losses, dill_file)

    test_loss = evaluate(model, test_iterator, loss_func)

    print(f'Test Loss: {test_loss[0]} | Train Accuracy: {test_loss[1]}%')

    translates = []
    for idx in range(9999):
        src = vars(test_data.examples[idx])['src']
        trg = vars(test_data.examples[idx])['trg']
        translation = translate_text(src, SRC, TRG, model, device)
        translates.append(' '.join(translation))
        print(idx)
    with open("outfile_ EMB_DIM = 256 HID_DIM = 256 ENC_LAYERS = 10 DEC_LAYERS = 10", "w") as outfile:
        outfile.write("\n".join(str(item) for item in translates))

