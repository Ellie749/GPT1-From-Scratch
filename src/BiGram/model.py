import torch
import torch.nn as nn
from torch.nn import functional as F


class BiGram(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x):
        return self.token_embedding_table(x)
    
    def calc_loss(self, logits, target):
        # logits = self(x)
        return F.cross_entropy(logits.view(-1, logits.shape[-1]), target.view(-1))

    @torch.no_grad()   
    def generate(self, x, max_new_tokens):
        for i in range(max_new_tokens):
            logits = self(x)
            probs = F.softmax(logits, dim=1)
            x_next = torch.multinomial(probs, 1, replacement=True)[0]
            x = torch.cat((x, x_next), dim=0)
    
        return x
    
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