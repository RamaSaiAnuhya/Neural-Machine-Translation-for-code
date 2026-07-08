import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, code_vocab_size, embedding_dim, hidden_dim, attention, pad_idx, num_layers=1, dropout = 0.1 ):
        super(Decoder, self).__init__()

        self.code_vocab_size = code_vocab_size
        self.hidden_dim = hidden_dim
        self.attention = attention
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(code_vocab_size, embedding_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size = embedding_dim + hidden_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first = True
        )
        self.output_layer = nn.Linear(hidden_dim, code_vocab_size)

    def forward(self, input_token, hidden, cell, encoder_outputs):
        input_token = input_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_token))
        context, attention_weights = self.attention(encoder_outputs, hidden[-1])
        context = context.unsqueeze(1)
        lstm_input = torch.cat((embedded, context), dim = 2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.output_layer(output.squeeze(1))

        return prediction, hidden, cell, attention_weights
