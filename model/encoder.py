import torch.nn as nn
from transformers import AutoModel


class BertEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased", freeze_bert=True):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        encoder_hidden_states = outputs.last_hidden_state
        return encoder_hidden_states, attention_mask


