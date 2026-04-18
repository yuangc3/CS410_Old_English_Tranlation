
import torch
import math
import torch.nn as nn
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos][i] = math.sin(pos / (10000 ** (i / d_model)))
                pe[pos][i+1]= math.cos(pos / (10000 ** (i / d_model)))
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        for i in range(x.size(0)):
           x[i] = x[i] + self.pe[i]
        return self.dropout(x)









    