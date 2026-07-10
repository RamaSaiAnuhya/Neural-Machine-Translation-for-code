import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super(BahdanauAttention, self).__init__()

        self.Wa = nn.Linear(decoder_hidden_dim, decoder_hidden_dim, bias=False)
        self.Ua = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim, bias=False)
        self.Va = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden, mask=None):
        decoder_hidden = decoder_hidden.unsqueeze(1)
        seq_len = encoder_outputs.shape[1]

        decoder_hidden_expanded = decoder_hidden.repeat(1, seq_len, 1)

        energy = torch.tanh(self.Wa(decoder_hidden_expanded) + self.Ua(encoder_outputs))

        scores = self.Va(energy)
        scores = scores.squeeze(2)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim = 1)
        attention_weights = attention_weights.unsqueeze(1)
        context = torch.bmm(attention_weights, encoder_outputs)
        
        attention_weights = attention_weights.squeeze(1)
        context = context.squeeze(1)

        return context, attention_weights
