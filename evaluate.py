import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import sacrebleu

from inference import load_model, translate

model, bert_tokenizer, target_tokenizer = load_model()

test_df = pd.read_csv("data/test.csv", encoding="utf-8")

predictions = []
references = []

for _, row in test_df.iterrows():
    src = str(row["english"]).strip()
    ref = str(row["old_english"]).strip()

    pred = translate(src, model, bert_tokenizer, target_tokenizer)
    predictions.append(pred)
    references.append(ref)

    print(f"SRC:  {src}")
    print(f"REF:  {ref}")
    print(f"PRED: {pred}")
    print()

bleu = sacrebleu.corpus_bleu(predictions, [references])
print(f"BLEU Score: {bleu.score:.2f}")
