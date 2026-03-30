import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sacrebleu
# Load data
train_df = pd.read_csv(r"C:\Users\caiyu\Desktop\cs410\data\train.csv")
val_df = pd.read_csv(r"C:\Users\caiyu\Desktop\cs410\data\val.csv")

# Fill missing values just in case
train_df["english"] = train_df["english"].fillna("")
train_df["old_english"] = train_df["old_english"].fillna("")
val_df["english"] = val_df["english"].fillna("")
val_df["old_english"] = val_df["old_english"].fillna("")

# Build TF-IDF on English training sentences
vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
train_tfidf = vectorizer.fit_transform(train_df["english"])

predictions = []
matched_train_sentences = []
similarity_scores = []

for sentence in val_df["english"]:
    val_vec = vectorizer.transform([sentence])
    sims = cosine_similarity(val_vec, train_tfidf)[0]
    best_idx = sims.argmax()

    pred_old_english = train_df.iloc[best_idx]["old_english"]
    matched_english = train_df.iloc[best_idx]["english"]
    best_score = sims[best_idx]

    predictions.append(pred_old_english)
    matched_train_sentences.append(matched_english)
    similarity_scores.append(best_score)

# Save results
results_df = val_df.copy()
results_df["retrieved_english"] = matched_train_sentences
results_df["prediction"] = predictions
results_df["similarity_score"] = similarity_scores

results_df.to_csv("val_retrieval_predictions.csv", index=False)

print(results_df[["english", "old_english", "retrieved_english", "prediction", "similarity_score"]].head(10))


results_df = pd.read_csv("val_retrieval_predictions.csv")


preds = results_df["prediction"].fillna("").tolist()
refs = [results_df["old_english"].fillna("").tolist()]

bleu = sacrebleu.corpus_bleu(preds, refs)
print("BLEU score:", bleu.score)
