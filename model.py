import random
import torch
import torch.nn as nn
from config import * 
import numpy as np
from dataset import Vocabulary
from abc import ABC, abstractmethod

class BiLSTM(torch.nn.Module):
    def __init__(self, src, trg, embedding_size, hidden_size):

        super().__init__()
        PAD = Vocabulary.PAD_TOKEN_IDX
        SRC_TOKEN_NUM = len(src.idx2word)
        TRG_TOKEN_NUM = len(trg.idx2word)

        src_dict = {"num_embeddings":SRC_TOKEN_NUM, "embedding_dim":embedding_size, "padding_idx":PAD}
        trg_dict = {"num_embeddings":TRG_TOKEN_NUM, "embedding_dim":embedding_size, "padding_idx":PAD}

        self.src_embedding = nn.Embedding(**src_dict)
        self.trg_embedding = nn.Embedding(**trg_dict)

        self.encoder = nn.LSTM(embedding_size, hidden_size, num_layers=1, bias=True, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTMCell(embedding_size, hidden_size * 2, bias=True)

        self.output = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Tanh(),
            nn.Dropout(DROP_OUT_DEFAULT),
            nn.Linear(hidden_size, TRG_TOKEN_NUM - 1)
        )

    def attention_forward(self,eh,em,dh,dm):
        batch_size, sequence_length, hidden_dim = eh.shape
        score = eh.bmm(dh.unsqueeze(-1)).squeeze(2)
        score[em] = (-np.inf)
        dist = torch.nn.Softmax(dim=1)(score)
        att = eh.transpose(1, 2).bmm(dist.unsqueeze(-1)).squeeze(2)
        att[dm, :] = 0.
        return att, dist

    def forward(self, src, trg_sentences, teacher_force=TEACHER_FORCE_DEFAULT):

        batch_size = src.shape[0]
        PAD = Vocabulary.PAD_TOKEN_IDX
        SOS = Vocabulary.SOS_TOKEN_IDX
        EOS = Vocabulary.EOS_TOKEN_IDX

        em = src == PAD

        src_embedding_seq = self.src_embedding(src)
        src_lengths = (src != 0).sum(dim=1)
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(src_embedding_seq, src_lengths.to('cpu'), batch_first=True, enforce_sorted=True)

        ehs, (hs, cell_state) = self.encoder(packed_seq)
        eh, _ = torch.nn.utils.rnn.pad_packed_sequence(ehs,
                                                                                 batch_first=True, padding_value=PAD)
        eh[em] = 0.

        dout = trg_sentences.new_full([batch_size], fill_value=SOS)
        dc0 = torch.cat((cell_state[0], cell_state[1]), dim=1)
        dh0 = torch.cat((hs[0], hs[1]), dim=1)
        sum_of_loss = 0.
        ce_loss = nn.CrossEntropyLoss(ignore_index=PAD, reduction='sum')
        for trg_word_idx in range(trg_sentences.shape[1] - 1):

            decoder_input = trg_sentences[:, trg_word_idx] if torch.distributions.bernoulli.Bernoulli(teacher_force).sample() else dout
            decoder_input_embedding = self.trg_embedding(decoder_input)
            dh0, dc0 = self.decoder(decoder_input_embedding, (dh0, dc0))
            decoder_mask = trg_sentences[:, trg_word_idx + 1] == PAD
            dh0[decoder_mask] = 0.

            aout, distribution = self.attention_forward(eh, em, dh0, decoder_mask)

            oul = torch.cat((dh0, aout), dim=1)
            ologit = self.output(oul)

            decoder_target = trg_sentences[:, trg_word_idx+1]
            dout = torch.argmax(ologit, dim=1) + 1
            tmp = ologit.new_full((batch_size, 1), fill_value=(-np.inf))
            new_logit = torch.cat((tmp, ologit), dim=1)
            loss = ce_loss(new_logit, decoder_target)
            sum_of_loss += loss

        loss = sum_of_loss
        assert loss.shape == torch.Size([])
        return loss / (trg_sentences[:, 1:] != PAD).sum()

    def translate(self, sentence, max_len=MAX_LEN_DEFAULT):
        PAD = Vocabulary.PAD_TOKEN_IDX
        SOS = Vocabulary.SOS_TOKEN_IDX
        EOS = Vocabulary.EOS_TOKEN_IDX

        translated =[]
        dist = []

        src_embedding_seq = self.src_embedding(sentence).unsqueeze(0)

        encoder_hidden_states, (hidden_state, cell_state) = self.encoder(src_embedding_seq)
        encoder_hidden=encoder_hidden_states

        dout = sentence.new_full([1], fill_value=SOS)
        dc0 = torch.cat((cell_state[0], cell_state[1]), dim=1)
        dh0 = torch.cat((hidden_state[0], hidden_state[1]), dim=1)
        encoder_masks = (sentence == PAD).unsqueeze(0)
        decoder_mask = False
        for trg_word_idx in range(max_len):
            decoder_input = dout
            decoder_input_embedding = self.trg_embedding(decoder_input)
            dh0, dc0 = self.decoder(decoder_input_embedding, (dh0, dc0))

            dh0[decoder_mask] = 0.
            aout, distribution = self.attention_forward(encoder_hidden, encoder_masks, dh0, decoder_mask)

            oul = torch.cat((dh0, aout), dim=1)
            ologit = self.output(oul)

            dout = torch.argmax(ologit, dim=1) + 1
            trans.append(dout)
            dist.append(distribution.squeeze(0))
            if dout == EOS:
                break
        trans = torch.stack(trans).squeeze(1)
        distributions = torch.stack(dist)

        return trans, dist
