import torch
import torch.nn as nn
from torchtyping import TensorType
import torch.nn.functional as F

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.Key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.Query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.Value = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.atn_dim = attention_dim


    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        K = self.Key(embedded)
        Q = self.Query(embedded)
        V = self.Value(embedded)

        seq_len = embedded.shape[1]

        atn_score = ((Q @ K.mT)/(self.atn_dim ** 0.5))

        mask = torch.tril(torch.ones(seq_len, seq_len))

        causal_attention = atn_score.masked_fill(mask == 0, float('-inf'))

        scores = F.softmax(causal_attention, dim=2)

        return torch.round(scores @ V, decimals=4)


        
