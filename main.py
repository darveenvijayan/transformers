import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embedding_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embedding_size
        self.heads = heads
        self.head_dim = embedding_size // heads

        assert (self.head_dim * heads == embedding_size), "embedding_size size needs to be a multiple of heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.key = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(self.heads*self.head_dim, embedding_size)

    def forward(self,)
