import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, text_vocab_size, embedding_dim, hidden_dim, pad_idx, num_layers=1, dropout=0.1):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(text_vocab_size, embedding_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc_hidden = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim*2, hidden_dim)

    def forward(self, src, src_lengths):
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False)

        packed_outputs, (hidden, cell) = self.lstm(packed)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

        # For 2 layers, hidden has shape (num_layers*2, batch, hidden_dim)
        # We want the last layer’s forward and backward states
        forward_hidden = hidden[-2, :, :]   # second-to-last = forward of last layer
        backward_hidden = hidden[-1, :, :]  # last = backward of last layer
        forward_cell = cell[-2, :, :]
        backward_cell = cell[-1, :, :]

        # Concatenate forward and backward
        hidden_cat = torch.cat((forward_hidden, backward_hidden), dim=1)
        cell_cat = torch.cat((forward_cell, backward_cell), dim=1)

        # Project down to hidden_dim
        decoder_hidden = torch.tanh(self.fc_hidden(hidden_cat)).unsqueeze(0)
        decoder_cell = torch.tanh(self.fc_cell(cell_cat)).unsqueeze(0)

        return outputs, decoder_hidden, decoder_cell