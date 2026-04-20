import torch
import torch.nn as nn
from model.encoder import BertEncoder
from model.decoder import TransformerDecoder
class Seq2SeqModel(nn.Module):
    def __init__(self, tgt_vocab_size, d_model = 256, nhead = 4, num_layers = 2, dim_feedforward = 512, dropout = 0.1):
        super().__init__()
        self.encoder = BertEncoder()
        self.enc_proj = nn.Linear(768, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.decoder = TransformerDecoder(d_model, nhead, num_layers, dim_feedforward, dropout)
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
        

    def forward(self, input_ids, attention_mask, tgt_input_ids):
        encoder_out, src_mask = self.encoder(input_ids, attention_mask)
        encoder_out = self.enc_proj(encoder_out)
        tgt_emb = self.tgt_embedding(tgt_input_ids)
        
        tgt_len = tgt_input_ids.size(1)
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(tgt_input_ids.device)
        decoder_out = self.decoder(tgt_emb, encoder_out, tgt_mask=tgt_mask, memory_mask=(src_mask == 0))
        logits = self.output_proj(decoder_out)
        return logits