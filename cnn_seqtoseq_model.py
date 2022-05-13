import torch
import torch.nn.functional as F


import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, kernel_size, dropout, device, max_length=300) -> None:
        super().__init__()
        self.device = device
        # normalizing the model to ensure the variance throughout the network does not change dramatically
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.token_embedding = nn.Embedding(input_dim, emb_dim)  # embedding each word
        self.pos_embedding = nn.Embedding(max_length, emb_dim)  # embedding position for each word in sequence

        self.embtohid = nn.Linear(emb_dim, hid_dim)
        self.hidtoemb = nn.Linear(hid_dim, emb_dim)
        self.conv_layers = nn.ModuleList([nn.Conv1d(in_channels=hid_dim,
                                                    out_channels=2 * hid_dim,
                                                    kernel_size=kernel_size,
                                                    padding=(kernel_size - 1) // 2)
                                          for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        batch_size = src.shape[0]
        src_len = src.shape[1]

        # create position tensor
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # embed value and positions
        tok_embedded = self.token_embedding(src)
        pos_embedded = self.pos_embedding(pos)
        embedded = self.dropout(tok_embedded + pos_embedded)
        conv_input = self.embtohid(embedded)
        conv_input = conv_input.permute(0, 2, 1)

        # passing through a list of convolution layers
        for i, conv_layer in enumerate(self.conv_layers):
            conv_output = conv_layer(self.dropout(conv_input))
            conv_output = F.glu(conv_output, dim=1)
            # add residual connection
            conv_output = (conv_output + conv_input) * self.scale
            conv_input = conv_output
        # change the dimension back to embedding dimension
        conv_output = self.hidtoemb(conv_output.permute(0, 2, 1))

        combined = (conv_output + embedded) * self.scale
        return conv_output, combined


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers,
                 kernel_size, dropout, trg_padding_index, device, max_length=300):
        super().__init__()
        self.kernel_size = kernel_size
        self.trg_padding_idx = trg_padding_index
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.token_embedding = nn.Embedding(output_dim, emb_dim)  # embedding each word
        self.pos_embedding = nn.Embedding(max_length, emb_dim)  # embedding position for each word

        self.embtohid = nn.Linear(emb_dim, hid_dim)
        self.hidtoemb = nn.Linear(hid_dim, emb_dim)
        self.attn_hidtoemb = nn.Linear(hid_dim, emb_dim)
        self.attn_embtohid = nn.Linear(emb_dim, hid_dim)
        self.fc_out = nn.Linear(emb_dim, output_dim)
        self.conv_layers = nn.ModuleList([nn.Conv1d(in_channels=hid_dim,
                                                    out_channels=2 * hid_dim,
                                                    kernel_size=kernel_size)
                                          for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def calculate_attention(self, embedded, conved, encoder_conved, encoder_combined):

        conved_emb = self.attn_hidtoemb(conved.permute(0, 2, 1))
        decoder_combined = (conved_emb + embedded) * self.scale
        # get multihead attention
        attention = F.softmax(torch.matmul(decoder_combined, encoder_conved.permute(0, 2, 1)), dim=2)
        encoder_with_attention = torch.matmul(attention, encoder_combined)
        encoder_with_attention = self.attn_embtohid(encoder_with_attention)
        conved_with_attention = (conved + encoder_with_attention.permute(0, 2, 1) * self.scale)
        return conved_with_attention

    def forward(self, trg, encoder_conved, encoder_combined):

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        tok_embedded = self.token_embedding(trg)
        pos_embedded = self.pos_embedding(pos)
        embedded = self.dropout(tok_embedded + pos_embedded)
        conv_input = self.embtohid(embedded)
        conv_input = conv_input.permute(0, 2, 1)
        hid_dim = conv_input.shape[1]
        for i, conv in enumerate(self.conv_layers):
            conv_input = self.dropout(conv_input)
            # padding so it won't see the predicted word
            padding = torch.zeros(batch_size,
                                  hid_dim,
                                  self.kernel_size - 1).fill_(self.trg_padding_idx).to(self.device)

            padding_conv_input = torch.cat((padding, conv_input), dim=2)
            # passing through convolutional layer
            conved_output = conv(padding_conv_input)
            # passing through GLU activation function
            conved_output = F.glu(conved_output, dim=1)
            # calculate multi-head attention
            conved_output = self.calculate_attention(embedded,
                                             conved_output,
                                             encoder_conved,
                                             encoder_combined)

            conved_output = (conved_output + conv_input) * self.scale
            conv_input = conved_output
        conved_output = self.hidtoemb(conved_output.permute(0, 2, 1))
        output = self.fc_out(self.dropout(conved_output))
        return output


class CnnSeqToSeq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        encoder_last_block, encoder_combined = self.encoder(src)
        output = self.decoder(trg, encoder_last_block, encoder_combined)
        return output