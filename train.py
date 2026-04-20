import torch
import torch.nn as nn
import pandas as pd
from functools import partial
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model.target_tokenizer import OldEnglishTokenizer
from model.data_utils import TranslationDataset, collate_fn
from model.model import Seq2SeqModel

# --- Config ---
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data ---
train_df = pd.read_csv("data/train.csv", encoding="utf-8")
val_df = pd.read_csv("data/val.csv", encoding="utf-8")

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

target_tokenizer = OldEnglishTokenizer(min_freq=1)
target_tokenizer.fit(train_df["old_english"].tolist())

train_dataset = TranslationDataset(train_df, bert_tokenizer, target_tokenizer)
val_dataset = TranslationDataset(val_df, bert_tokenizer, target_tokenizer)

collate = partial(collate_fn, src_pad_id=bert_tokenizer.pad_token_id, tgt_pad_id=target_tokenizer.pad_id)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

# --- Model ---
vocab_size = len(target_tokenizer.token_to_id)
model = Seq2SeqModel(tgt_vocab_size=vocab_size).to(DEVICE)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=target_tokenizer.pad_id)

# --- Training Loop ---
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        src_input_ids = batch["src_input_ids"].to(DEVICE)
        src_attention_mask = batch["src_attention_mask"].to(DEVICE)
        tgt_input_ids = batch["tgt_input_ids"].to(DEVICE)

        # decoder input: <bos> + sequence (drop last token)
        tgt_in = tgt_input_ids[:, :-1]
        # decoder target: sequence + <eos> (drop first token)
        tgt_out = tgt_input_ids[:, 1:]

        logits = model(src_input_ids, src_attention_mask, tgt_in)

        # logits: [batch, tgt_len, vocab_size] → [batch*tgt_len, vocab_size]
        loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # --- Validation ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            src_input_ids = batch["src_input_ids"].to(DEVICE)
            src_attention_mask = batch["src_attention_mask"].to(DEVICE)
            tgt_input_ids = batch["tgt_input_ids"].to(DEVICE)

            tgt_in = tgt_input_ids[:, :-1]
            tgt_out = tgt_input_ids[:, 1:]

            logits = model(src_input_ids, src_attention_mask, tgt_in)
            loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

torch.save(model.state_dict(), "model_weights.pt")
print("Model saved to model_weights.pt")
