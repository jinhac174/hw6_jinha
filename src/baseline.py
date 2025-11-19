import re
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report


keywords = {
    "business": ["market", "trade", "shares", "profit", "company", "finance"],
    "entertainment": ["film", "movie", "music", "show", "celebrity"],
    "politics": ["election", "government", "minister", "parliament", "policy"],
    "sport": ["match", "team", "win", "goal", "tournament", "league"],
    "tech": ["technology", "software", "device", "internet", "computer"],
}

labels = ["business", "entertainment", "politics", "sport", "tech"]
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

def predict(text):
    text_l = text.lower()
    scores = {l: 0 for l in labels}
    for cat, words in keywords.items():
        for w in words:
            if w in text_l:
                scores[cat] += 1

    best = max(scores, key=scores.get)
    return label2id[best]


print("Loading BBC News dataset...")
dataset = load_dataset("SetFit/bbc-news")
test_texts = dataset["test"]["text"]
test_labels = dataset["test"]["label"]

preds = [predict(t) for t in test_texts]

acc = accuracy_score(test_labels, preds)
print(f"\n=== BASELINE BBC RESULTS ===")
print(f"Accuracy: {acc:.3f}\n")
print
