"""
This script runs the full pipeline:
  1. Train the model (if no weights found)
  2. Evaluate on the test set (BLEU score)
  3. Run demo translations
Command:
    python run.py
"""

import sys
import io
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import torch
import pandas as pd
from functools import partial
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import sacrebleu

from model.target_tokenizer import OldEnglishTokenizer
from model.data_utils import TranslationDataset, collate_fn
from model.model import Seq2SeqModel
from model.encoder import BertEncoder
import torch.nn as nn

# ── Config ──────────────────────────────────────────────────────────────────
WEIGHTS_PATH = "model_weights.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-3
MAX_TGT_LEN = 64

# ── Load data & tokenizers ───────────────────────────────────────────────────
print("=" * 60)
print("CS410 Final Project: Modern English → Old English Translation")
print("=" * 60)

train_df = pd.read_csv("data/train.csv", encoding="utf-8")
val_df   = pd.read_csv("data/val.csv",   encoding="utf-8")
test_df  = pd.read_csv("data/test.csv",  encoding="utf-8")

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

target_tokenizer = OldEnglishTokenizer(min_freq=1)
target_tokenizer.fit(train_df["old_english"].tolist())
vocab_size = len(target_tokenizer.token_to_id)

print(f"\nDataset: {len(train_df)} train | {len(val_df)} val | {len(test_df)} test")
print(f"Old English vocabulary size: {vocab_size}")

# ── Train if no weights found ────────────────────────────────────────────────
if not os.path.exists(WEIGHTS_PATH):
    print(f"\nNo weights found at '{WEIGHTS_PATH}'. Training from scratch...")

    collate = partial(collate_fn,
                      src_pad_id=bert_tokenizer.pad_token_id,
                      tgt_pad_id=target_tokenizer.pad_id)

    train_loader = DataLoader(
        TranslationDataset(train_df, bert_tokenizer, target_tokenizer),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(
        TranslationDataset(val_df, bert_tokenizer, target_tokenizer),
        batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    torch.manual_seed(42)
    model = Seq2SeqModel(tgt_vocab_size=vocab_size).to(DEVICE)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = nn.CrossEntropyLoss(
        ignore_index=target_tokenizer.pad_id, label_smoothing=0.1)

    best_val_loss = float("inf")
    patience, no_improve = 10, 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            src_ids  = batch["src_input_ids"].to(DEVICE)
            src_mask = batch["src_attention_mask"].to(DEVICE)
            tgt_ids  = batch["tgt_input_ids"].to(DEVICE)
            tgt_in, tgt_out = tgt_ids[:, :-1], tgt_ids[:, 1:]
            logits = model(src_ids, src_mask, tgt_in)
            loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                src_ids  = batch["src_input_ids"].to(DEVICE)
                src_mask = batch["src_attention_mask"].to(DEVICE)
                tgt_ids  = batch["tgt_input_ids"].to(DEVICE)
                tgt_in, tgt_out = tgt_ids[:, :-1], tgt_ids[:, 1:]
                logits = model(src_ids, src_mask, tgt_in)
                val_loss += criterion(
                    logits.reshape(-1, vocab_size), tgt_out.reshape(-1)).item()

        avg_train = total_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            no_improve = 0
            torch.save(model.state_dict(), WEIGHTS_PATH)
            print("  → Best model saved!")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
else:
    print(f"\nFound existing weights at '{WEIGHTS_PATH}'. Skipping training.")

# ── Load model ───────────────────────────────────────────────────────────────
model = Seq2SeqModel(tgt_vocab_size=vocab_size).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.eval()


# ── Translate helper ─────────────────────────────────────────────────────────
def translate(sentence):
    encoding = bert_tokenizer(sentence, return_tensors="pt",
                              truncation=True, max_length=64)
    src_ids  = encoding["input_ids"].to(DEVICE)
    src_mask = encoding["attention_mask"].to(DEVICE)
    generated = [target_tokenizer.bos_id]

    with torch.no_grad():
        encoder_out, enc_mask = model.encoder(src_ids, src_mask)
        encoder_out  = model.enc_proj(encoder_out)
        memory_mask  = (enc_mask == 0)

        for _ in range(MAX_TGT_LEN):
            tgt     = torch.tensor([generated], dtype=torch.long).to(DEVICE)
            tgt_emb = model.tgt_embedding(tgt)
            tgt_len = tgt.size(1)
            tgt_mask = torch.triu(
                torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(DEVICE)
            dec_out = model.decoder(tgt_emb, encoder_out,
                                    tgt_mask=tgt_mask, memory_mask=memory_mask)
            next_token = model.output_proj(dec_out)[0, -1, :].argmax().item()
            if next_token == target_tokenizer.eos_id:
                break
            generated.append(next_token)

    return target_tokenizer.decode(generated, skip_special_tokens=True)


# ── BLEU evaluation on test set ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("Evaluating on test set...")
print("=" * 60)

predictions, references = [], []
for _, row in test_df.iterrows():
    pred = translate(str(row["english"]).strip())
    predictions.append(pred)
    references.append(str(row["old_english"]).strip())

bleu = sacrebleu.corpus_bleu(predictions, [references])
print(f"\nTest BLEU Score: {bleu.score:.2f}")

# ── Demo translations ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Demo Translations")
print("=" * 60)

demo_sentences = [
    "God is everywhere.",
    "The king went to the hall.",
    "I do not know him.",
    "In the beginning God created the heaven and the earth.",
    "To be, or not to be, that is the question.",
]

for sentence in demo_sentences:
    result = translate(sentence)
    print(f"Modern English : {sentence}")
    print(f"Old English    : {result}")
    print()
