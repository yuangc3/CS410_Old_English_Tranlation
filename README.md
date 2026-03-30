# Modern English to Old English Translation

## Project Overview
This project explores automatic translation from Modern English to Old English. Our goal is to study how well simple and more advanced NLP methods can map modern sentence forms into historically older linguistic forms. At the current milestone stage, we have completed dataset cleaning, exploratory data analysis, train/validation/test splitting, and an initial retrieval-based baseline.

## Current Progress
- Cleaned and organized the dataset
- Performed exploratory data analysis
- Split the dataset into train, validation, and test sets
- Implemented a TF-IDF retrieval baseline
- Evaluated the baseline on the validation set using BLEU

## Dataset Summary
The cleaned dataset currently contains **404 sentence pairs**.

Example pairs:
- **English:** `that was the best day of my life.`  
  **Old English:** `þæt ƿæs se besta dæᵹ mīnes līfes.`
- **English:** `i don't want to go to school.`  
  **Old English:** `ic nelle to þæm larhuse.`
- **English:** `freedom is not free.`  
  **Old English:** `friþ-dōm ne is un-cēap.`

Basic statistics:
- Total pairs: **404**
- Average English sentence length: **6.13**
- Average Old English sentence length: **5.85**
- English vocabulary size: **920**
- Old English vocabulary size: **1121**

## Project Structure
```text
project/
│
├── data/
│   ├── cleaned/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── model/
│   └── baseline.py
│
│
├── report/
│   └── milestone_report.tex
│
└── README.md
