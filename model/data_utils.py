import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

class TranslationDataset:
    def __init__(self, dataframe, bert_tokenizer, target_tokenizer, max_src_len= 64, max_tgt_len=64):
        self.dataframe = dataframe
        self.bert_tokenizer = bert_tokenizer
        self.target_tokenizer = target_tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
    
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        src_text = str(row["english"]).strip()
        tgt_text = str(row["old_english"]).strip()
        src_encoding = self.bert_tokenizer(src_text, add_special_tokens=True, max_length=self.max_src_len, truncation=True, return_tensors="pt")
        tgt_ids = self.target_tokenizer.encode(tgt_text, add_special_tokens=True)
        tgt_ids = tgt_ids[:self.max_tgt_len]
        return {
    "src_input_ids": src_encoding["input_ids"].squeeze(0),
    "src_attention_mask": src_encoding["attention_mask"].squeeze(0),
    "tgt_input_ids": torch.tensor(tgt_ids, dtype=torch.long),
}


def collate_fn(batch, src_pad_id, tgt_pad_id):
    src_input_ids = []
    src_attention_masks = []
    tgt_input_ids = []
    for item in batch:
        src_input_ids.append(item["src_input_ids"])
        src_attention_masks.append(item["src_attention_mask"])
        tgt_input_ids.append(item["tgt_input_ids"])
    
    src_input_ids = pad_sequence(src_input_ids, batch_first=True, padding_value=src_pad_id)
    src_attention_masks = pad_sequence(src_attention_masks, batch_first=True, padding_value=0)
    tgt_input_ids = pad_sequence(tgt_input_ids, batch_first=True, padding_value=tgt_pad_id)

    return {
        "src_input_ids": src_input_ids,
        "src_attention_mask": src_attention_masks,
        "tgt_input_ids": tgt_input_ids,
    }




