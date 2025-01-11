import torch
import torch.nn as nn


class Head(nn.Module):
    def __init__(self, n_heads, embedding_dim, head_size, sequence_length):
        super(nn.Module).__init__()
        self.n_heads = n_heads
        self.w_key = nn.Linear(embedding_dim, head_size)
        self.w_query = nn.Linear(embedding_dim, head_size)
        self.w_value = nn.Linear(embedding_dim, head_size)
        self.register_buffer('trill', torch.tril(torch.ones(sequence_length, sequence_length)))


    def forward(self, x):
        Q = self.w_query(x)
        K = self.w_key(x)
        V = self.w_value(x)

        l = Q