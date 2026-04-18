import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from functools import partial
from model.data_utils import TranslationDataset, collate_fn

from model.target_tokenizer import OldEnglishTokenizer
from model.data_utils import TranslationDataset

df = pd.read_csv("data/train.csv", encoding="utf-8")

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

target_tokenizer = OldEnglishTokenizer(min_freq=1)
target_tokenizer.fit(df["old_english"].tolist())

dataset = TranslationDataset(df, bert_tokenizer, target_tokenizer)
sample = dataset[0]

print(sample.keys())
print(sample["src_input_ids"].shape)
print(sample["src_attention_mask"].shape)
print(sample["tgt_input_ids"].shape)

#test the dataloader and collate_fn
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

print(batch["src_input_ids"].shape)
print(batch["src_attention_mask"].shape)
print(batch["tgt_input_ids"].shape)
