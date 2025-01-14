import torch
import torch.nn as nn
from torch.nn import functional as F
#add dropout

class Head(nn.Module):
    def __init__(self, embedding_dim, head_size, sequence_length):
        super().__init__()
        self.head_size = head_size
        self.w_key = nn.Linear(embedding_dim, head_size)
        self.w_query = nn.Linear(embedding_dim, head_size)
        self.w_value = nn.Linear(embedding_dim, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(sequence_length, sequence_length))) # since it is not a PyTorch parameter
        # Since buffers are not parameters, they do not consume memory in the 
        # computational graph and do not contribute to gradient computation.
        # This avoids unnecessary computation and potential errors.

    def forward(self, x):
        Q = self.w_query(x)
        K = self.w_key(x)
        V = self.w_value(x)
        wei = Q @ K.transpose(-2, -1) / self.head_size**0.5
        wei = wei.masked_fill(self.tril==0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # default of dim is -1 but implicit dimension choice is deprecated
        out = wei @ V

        return out
    

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, heads_dim, embedding_dim, sequence_length):
        super().__init__()
        self.attention_weight = nn.Linear(heads_dim*n_heads, embedding_dim)
        self.heads = nn.ModuleList([Head(embedding_dim, heads_dim, sequence_length) for _ in range(n_heads)])

    def forward(self, x):
        mha = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.attention_weight(mha)