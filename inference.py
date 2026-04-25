import torch
import pandas as pd
from transformers import AutoTokenizer

from model.target_tokenizer import OldEnglishTokenizer
from model.model import Seq2SeqModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_TGT_LEN = 64


def load_model(weights_path="model_weights.pt"):
    train_df = pd.read_csv("data/train.csv", encoding="utf-8")

    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    target_tokenizer = OldEnglishTokenizer(min_freq=1)
    target_tokenizer.fit(train_df["old_english"].tolist())

    vocab_size = len(target_tokenizer.token_to_id)
    model = Seq2SeqModel(tgt_vocab_size=vocab_size).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()

    return model, bert_tokenizer, target_tokenizer


def translate(sentence, model, bert_tokenizer, target_tokenizer):
    # Encode source sentence
    encoding = bert_tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=64,
    )
    src_input_ids = encoding["input_ids"].to(DEVICE)
    src_attention_mask = encoding["attention_mask"].to(DEVICE)

    # Greedy decode
    generated = [target_tokenizer.bos_id]

    with torch.no_grad():
        encoder_out, src_mask = model.encoder(src_input_ids, src_attention_mask)
        encoder_out = model.enc_proj(encoder_out)
        memory_mask = (src_mask == 0)

        for _ in range(MAX_TGT_LEN):
            tgt = torch.tensor([generated], dtype=torch.long).to(DEVICE)
            tgt_emb = model.tgt_embedding(tgt)

            tgt_len = tgt.size(1)
            tgt_mask = torch.triu(
                torch.ones(tgt_len, tgt_len), diagonal=1
            ).bool().to(DEVICE)

            decoder_out = model.decoder(tgt_emb, encoder_out,
                                        tgt_mask=tgt_mask, memory_mask=memory_mask)
            logits = model.output_proj(decoder_out)

            next_token = logits[0, -1, :].argmax().item()
            if next_token == target_tokenizer.eos_id:
                break
            generated.append(next_token)

    return target_tokenizer.decode(generated, skip_special_tokens=True)


if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    model, bert_tokenizer, target_tokenizer = load_model()

    test_sentences = [
        "I love English.",
        "The king went to the hall.",
        "She loves him.",
        "God is everywhere.",
        "I do not know him.",
    ]

    for sentence in test_sentences:
        result = translate(sentence, model, bert_tokenizer, target_tokenizer)
        print(f"Modern:  {sentence}")
        print(f"Old English: {result}")
        print()
