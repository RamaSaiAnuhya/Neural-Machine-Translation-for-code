import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, text_vocab_size, embedding_dim, hidden_dim, pad_idx, num_layers=1, dropout=0.1):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(text_vocab_size, embedding_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0, batch_first=True)


    def forward(self, src, src_lengths):
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False)

        packed_outputs, (hidden, cell) = self.lstm(packed)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

        return outputs, hidden, cell