import requests
import pandas as pd
import time
from sklearn.model_selection import train_test_split

EXISTING_DATA = "data/cleaned_dataset.csv"


def fetch_tatoeba_page(page):
    url = "https://tatoeba.org/en/api_v0/search"
    params = {"from": "eng", "to": "ang", "query": "", "page": page}
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, params=params, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()


def extract_pairs(data):
    pairs = []
    for result in data.get("results", []):
        english = result.get("text", "").strip()
        for translation in result.get("translations", []):
            for t in translation:
                if t.get("lang") == "ang":
                    old_english = t.get("text", "").strip()
                    if english and old_english:
                        pairs.append({"english": english, "old_english": old_english})
    return pairs


def main():
    existing_df = pd.read_csv(EXISTING_DATA, encoding="utf-8")
    existing_pairs = set(zip(existing_df["english"].str.strip(), existing_df["old_english"].str.strip()))
    print(f"Existing pairs: {len(existing_df)}")

    new_pairs = []
    page = 1

    while True:
        print(f"Fetching page {page}...")
        try:
            data = fetch_tatoeba_page(page)
        except Exception as e:
            print(f"Error: {e}")
            break

        pairs = extract_pairs(data)
        if not pairs:
            print("No more results.")
            break

        for pair in pairs:
            key = (pair["english"], pair["old_english"])
            if key not in existing_pairs:
                new_pairs.append(pair)
                existing_pairs.add(key)

        total_pages = data.get("paging", {}).get("Sentences", {}).get("pageCount", 1)
        print(f"  Page {page}/{total_pages} — {len(pairs)} pairs")

        if page >= total_pages:
            break
        page += 1
        time.sleep(0.5)

    print(f"\nNew pairs found: {len(new_pairs)}")

    if not new_pairs:
        print("No new data. Exiting.")
        return

    # Merge with existing and re-split
    new_df = pd.DataFrame(new_pairs)
    combined_df = pd.concat([existing_df[["english", "old_english"]], new_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates().reset_index(drop=True)
    print(f"Total pairs: {len(combined_df)}")

    combined_df.to_csv(EXISTING_DATA, index=False, encoding="utf-8")

    # Re-split into train/val/test (80/10/10)
    train_df, temp_df = train_test_split(combined_df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_df.to_csv("data/train.csv", index=False, encoding="utf-8")
    val_df.to_csv("data/val.csv", index=False, encoding="utf-8")
    test_df.to_csv("data/test.csv", index=False, encoding="utf-8")

    print(f"train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    print("Done!")


if __name__ == "__main__":
    main()
