import torch
import torch.nn as nn
from torch.nn import functional as F
from attention import MultiHeadAttention
from ffn import FFN


class Block(nn.Module):
    def __init__(self, embedding_dim, sequence_length, n_heads, heads_dim):
        super().__init__()
        self.heads = MultiHeadAttention(n_heads, heads_dim, embedding_dim, sequence_length)
        self.ffn = FFN(embedding_dim)
        self.normlayeratt = nn.LayerNorm(embedding_dim)
        self.normlayerffn = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x + self.heads(self.normlayeratt(x))
        out = x + self.ffn(self.normlayerffn(x))

        return out


class GPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, sequence_length, n_heads, heads_dim, n_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.position_embeddings = nn.Embedding(sequence_length, embedding_dim)
        self.layers = nn.ModuleList([Block(embedding_dim, sequence_length, n_heads, heads_dim) for _ in range(n_layers)])
        # self.layers = [Block(embedding_dim, sequence_length, n_heads, heads_dim) for _ in range(n_layers)]

        self.norm = nn.LayerNorm(embedding_dim)
        self.language_model = nn.Linear(embedding_dim, vocab_size)


    def forward(self, x):
        e = self.token_embeddings(x) + self.position_embeddings(torch.arange(x.shape[-1], device=x.device))
        for layer in self.layers:
            e = layer(e)
        logits = self.language_model(self.norm(e))

        return logits


    def calc_loss(self, logits, target):
        return F.cross_entropy(logits.view(-1, logits.shape[-1]), target.view(-1))


    def train(self, X_train, y_train, X_validation, y_validation, epochs=10):
        history = []
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        epoch_loss = 10
        for i in range(epochs):
            print(f"[INFO]  epoch {i+1}, train_loss: {epoch_loss/X_train.shape[0]}, validation_loss: {self.calc_loss(self(X_validation), y_validation)}")
            
            epoch_loss = 0
            for b in range(X_train.shape[0]):
                logits = self(X_train[b])
                loss = self.calc_loss(logits, y_train[b])
                epoch_loss+=loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                # print(f"loss: {loss.item()}")

            history.append(epoch_loss.item()/X_train.shape[0])
        
        return history

    def inference(self, x, max_new_tokens, sequence):
        for i in range(max_new_tokens):
            x_cond = x[:self.sequence_length]
            logits = self(x_cond)[:, -1, :]
            probs = F.softmax(logits)
            next = torch.multinomial(probs, 1, replacement=True)

            x = torch.cat((x, next), dim=1)

        return x