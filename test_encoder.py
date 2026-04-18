import pandas as pd
from functools import partial
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model.target_tokenizer import OldEnglishTokenizer
from model.data_utils import TranslationDataset, collate_fn
from model.encoder import BertEncoder

df = pd.read_csv("data/train.csv", encoding="utf-8")

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

target_tokenizer = OldEnglishTokenizer(min_freq=1)
target_tokenizer.fit(df["old_english"].tolist())

dataset = TranslationDataset(df, bert_tokenizer, target_tokenizer)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=partial(
        collate_fn,
        src_pad_id=bert_tokenizer.pad_token_id,
        tgt_pad_id=target_tokenizer.pad_id,
    ),
)

batch = next(iter(loader))

encoder = BertEncoder()
encoder_hidden_states, encoder_attention_mask = encoder(
    batch["src_input_ids"],
    batch["src_attention_mask"],
)

print(batch["src_input_ids"].shape)
print(batch["src_attention_mask"].shape)
print(encoder_hidden_states.shape)
print(encoder_attention_mask.shape)
