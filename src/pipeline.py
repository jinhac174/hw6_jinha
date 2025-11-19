import os
import random
import numpy as np
import torch
import torch.nn as nn

from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

# -------------------- config --------------------
model_name = "distilbert-base-uncased"
num_labels = 5
max_length = 256
batch_size = 16
epochs = 3
lr = 2e-5
warmup_ratio = 0.1
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(seed)

# -------------------- data --------------------
dataset = load_dataset("SetFit/bbc-news")
train_df = dataset["train"].to_pandas()
test_df = dataset["test"].to_pandas()

label_names = ["business", "entertainment", "politics", "sport", "tech"]

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df["text"].tolist(),
    train_df["label"].tolist(),
    test_size=0.1,
    random_state=seed,
    stratify=train_df["label"].tolist(),
)

test_texts = test_df["text"].tolist()
test_labels = test_df["label"].tolist()

tokenizer = AutoTokenizer.from_pretrained(model_name)


class bbc_dataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


train_dataset = bbc_dataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = bbc_dataset(val_texts, val_labels, tokenizer, max_length)
test_dataset = bbc_dataset(test_texts, test_labels, tokenizer, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# -------------------- model --------------------
class classifier_head_linear(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.out(x)


class classifier_head_mlp(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_labels),
        )

    def forward(self, x):
        return self.net(x)


class bbc_classifier(nn.Module):
    def __init__(self, model_name, num_labels, head_type):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        if head_type == "linear":
            self.head = classifier_head_linear(hidden_size, num_labels)
        else:
            self.head = classifier_head_mlp(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.head(cls)


# -------------------- loss --------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1, 2, 3, 4]),
    y=np.array(train_labels),
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)


def get_loss_fn(weighted=False):
    if weighted:
        return nn.CrossEntropyLoss(weight=class_weights_tensor)
    return nn.CrossEntropyLoss()


# -------------------- train --------------------
def train_one_epoch(model, loader, optimizer, scheduler, loss_fn):
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    preds, trues = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            total_loss += loss.item()
            pred = torch.argmax(logits, dim=-1)

            preds += pred.cpu().tolist()
            trues += labels.cpu().tolist()

    acc = accuracy_score(trues, preds)
    return total_loss / len(loader), acc, trues, preds


# -------------------- ablation variants --------------------
variants = [
    {"name": "v1_linear_ce", "head": "linear", "weighted": False},
    {"name": "v2_linear_weighted", "head": "linear", "weighted": True},
    {"name": "v3_mlp_ce", "head": "mlp", "weighted": False},
    {"name": "v4_mlp_weighted", "head": "mlp", "weighted": True},
]

results = []

# -------------------- run ablations --------------------
for cfg in variants:
    print(f"\n===== training {cfg['name']} =====")

    model = bbc_classifier(model_name, num_labels, head_type=cfg["head"]).to(device)
    loss_fn = get_loss_fn(cfg["weighted"])

    steps_total = epochs * len(train_loader)
    warmup_steps = int(warmup_ratio * steps_total)

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=steps_total
    )

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, loss_fn)
        print(f"epoch {epoch} | train {train_loss:.4f} | val {val_loss:.4f} | acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # load best val model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # final test performance
    _, test_acc, true_labels, pred_labels = evaluate(model, test_loader, loss_fn)
    print(f"test acc: {test_acc:.4f}")
    print(classification_report(true_labels, pred_labels, target_names=label_names))

    results.append(
        {
            "name": cfg["name"],
            "head": cfg["head"],
            "weighted": cfg["weighted"],
            "val_acc": best_val_acc,
            "test_acc": test_acc,
        }
    )

# -------------------- summary --------------------
print("\n===== ablation summary =====")
print(f"{'variant':25s} {'head':10s} {'weighted':10s} {'val_acc':10s} {'test_acc':10s}")

for r in results:
    print(
        f"{r['name']:25s} {r['head']:10s} {str(r['weighted']):10s} "
        f"{r['val_acc']:.4f}     {r['test_acc']:.4f}"
    )
