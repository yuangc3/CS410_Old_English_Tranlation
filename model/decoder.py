
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
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)



class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward = 512, dropout=0.1):
        super().__init__()#initialize the decoder layer with multi-head self-attention, multi-head cross-attention, and feedforward network
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True) #multihead self-attentinon
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True) #multihead cross-attention
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )#activation function for the feedforward network
        self.norm1 = nn.LayerNorm(d_model)#Norm
        self.norm2 = nn.LayerNorm(d_model)#Norm
        self.norm3 = nn.LayerNorm(d_model)#Norm
        self.dropout = nn.Dropout(dropout)#Dropout
    
    def forward(self, x, encoder_output, tgt_mast =None , memory_mask = None):
        attn_output, _ = self.self_attn(x, x, x, attn_mask = tgt_mast) #self attention
        x = self.norm1(x+ self.dropout(attn_output))#Norm
        attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, key_padding_mask = memory_mask) #cross attention
        x = self.norm2(x + self.dropout(attn_output))#Norm
        ff_output = self.ff(x) #feedforward network
        x = self.norm3(x+ self.dropout(ff_output))#Norm
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, d_model = 256, nhead = 4, num_layers = 2, dim_feedforward =512, dropout = 0.1):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout) #positional encoding
        self.layers = nn.ModuleList([DecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)]) #decoder layers
    
    def forward(self, tgt, encoder_output, tgt_mask = None, memory_mask = None):
        x = self.pos_enc(tgt)
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask = tgt_mask, memory_mask = memory_mask)
        return x




    