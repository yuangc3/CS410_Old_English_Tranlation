# Modern English to Old English Translation

## Project Overview
This project builds a neural machine translation system that translates Modern English sentences into Old English. We implement a hybrid transformer architecture combining a pretrained BERT encoder with a custom Transformer decoder, and compare it against a TF-IDF retrieval baseline.

## Model Architecture
```
Modern English → BERT Encoder → Linear Projection → Transformer Decoder → Old English
```
- **Encoder:** `bert-base-uncased` (frozen) — produces contextual embeddings
- **Decoder:** 2-layer Transformer with masked self-attention and cross-attention
- **Tokenizer:** BERT tokenizer for source, custom word-level tokenizer for target

## Results

| Model | BLEU Score |
|-------|-----------|
| TF-IDF Retrieval Baseline | 12.0 |
| BERT + Transformer Decoder | **63.48** |

## Dataset
- **Source:** Tatoeba (English ↔ Old English sentence pairs)
- **Total pairs:** 1191
- **Split:** 80% train / 10% val / 10% test

Example pairs:
- **English:** `that was the best day of my life.`
  **Old English:** `þæt ƿæs se besta dæᵹ mīnes līfes.`
- **English:** `god is everywhere.`
  **Old English:** `god biþ ǣġhwǣr.`

## Project Structure
```
CS410_Old_English_Translation/
├── data/
│   ├── cleaned_dataset.csv
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── model/
│   ├── encoder.py           # BERT encoder wrapper
│   ├── decoder.py           # Transformer decoder
│   ├── model.py             # Full Seq2Seq model
│   ├── data_utils.py        # Dataset and collate functions
│   ├── target_tokenizer.py  # Old English tokenizer
│   └── baseline.py          # TF-IDF retrieval baseline
├── train.py                 # Training loop with early stopping
├── inference.py             # Greedy decoding inference
├── evaluate.py              # BLEU evaluation
└── fetch_data.py            # Tatoeba data fetching script
```

## How to Run

**Full pipeline (train if needed + evaluate + demo):**
```bash
python run.py
```

**Train only:**
```bash
python train.py
```

**Translate a sentence:**
```bash
python inference.py
```

**Evaluate BLEU on test set:**
```bash
python -X utf8 evaluate.py
```

**Fetch more data from Tatoeba:**
```bash
python fetch_data.py
```
