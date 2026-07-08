import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, src_lengths, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        tgt_length = tgt.shape[1]
        output_dim = self.decoder.code_vocab_size

        outputs = torch.empty(batch_size, tgt_length, output_dim, device=self.device)

        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)

        input_token = tgt[:, 0]

        for t in range(1, tgt_length):
            prediction, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_outputs)
            outputs[:, t] = prediction
            teacher_force = random.random() < teacher_forcing_ratio
            predicted_token = prediction.argmax(1)
            input_token = (tgt[:, t] if teacher_force else predicted_token)

        return outputs

