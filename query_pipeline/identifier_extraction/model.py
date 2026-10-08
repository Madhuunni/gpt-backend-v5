import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class IDTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim=48, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.15)
        self.classifier = nn.Linear(2*hidden_dim, 4)

    def forward(self, tokens, lengths):
        packed = pack_padded_sequence(self.embedding(tokens), lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.encoder(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=tokens.shape[1])
        return self.classifier(self.dropout(encoded))
