import json, math, os, random, argparse
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.optim as optim
from transformers import BertConfig, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

RANDOM_SEED = 42
MAX_LEN      = 256
BATCH_SIZE   = 16
EPOCHS       = 4
LR           = 2e-5
VOCAB_SIZE   = 17120

torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def read_json(path: Path, with_label: bool = True):
    data = []
    for obj in map(json.loads, path.read_text().splitlines()):
        ids  = obj["text"][:MAX_LEN]
        pad  = [0] * (MAX_LEN - len(ids))
        item = {
            "input_ids": torch.tensor(ids + pad, dtype=torch.long),
            "attention_mask": torch.tensor([1]*len(ids) + pad, dtype=torch.long)
        }
        if with_label:
            item["labels"] = torch.tensor(obj["label"], dtype=torch.long)
        data.append(item)
    return data


class JsonDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


def build_model():
    cfg = BertConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=MAX_LEN,
        type_vocab_size=1,
        num_labels=2
    )
    model = BertForSequenceClassification(cfg)
    return model


def train_loop(model, loader, optimiser, scheduler, device, criterion):
    model.train()
    total, correct = 0, 0
    for batch in tqdm(loader, desc="train"):
        labels = batch.pop("labels").to(device)
        batch  = {k: v.to(device) for k, v in batch.items()}

        optimiser.zero_grad()
        logits = model(**batch).logits
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        scheduler.step()

        preds = logits.argmax(-1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    return correct / total



@torch.no_grad()
def eval_loop(model, loader, device):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}

        logits = model(**batch).logits
        ys.extend(labels.cpu().tolist())
        ps.extend(logits.argmax(-1).cpu().tolist())
    acc = accuracy_score(ys, ps)
    f1  = f1_score(ys, ps, average="macro")
    return acc, f1


def main(args):
    # read data ──────────────────────────────────────────────
    d1 = read_json(Path("domain1_train_data.json"))
    d2 = read_json(Path("domain2_train_data.json"))
    random.shuffle(d1); random.shuffle(d2)

    split1 = int(0.9 * len(d1))
    split2 = int(0.9 * len(d2))
    train_ds = JsonDataset(d1[:split1] + d2[:split2])
    val_ds   = JsonDataset(d1[split1:] + d2[split2:])
    test_ds  = JsonDataset(read_json(Path("test_data.json"), with_label=False))

    # Imbalanced‑data handling ──────────────────────────────────────────────
    label_list = [s["labels"].item() for s in train_ds]  # 0 / 1
    class_counts = torch.bincount(torch.tensor(label_list))  # e.g. tensor([4200,  800])
    print("train class counts:", class_counts.tolist())

    weights = 1.0 / class_counts[label_list]
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(train_ds),
        replacement=True
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        drop_last=False
    )

    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    # build model ──────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model().to(device)

    optimiser = optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_dl) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimiser, num_warmup_steps=int(0.1*total_steps),
        num_training_steps=total_steps)

    class_weights = (1.0 / class_counts).to(device)
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        reduction="mean"
    )

    # train model ────────────────────────────────────────────────
    best_f1 = 0.0
    for ep in range(1, EPOCHS+1):
        tr_acc = train_loop(model, train_dl, optimiser, scheduler, device, criterion)
        val_acc, val_f1 = eval_loop(model, val_dl, device)
        print(f"Epoch {ep}: train acc={tr_acc:.4f} | val acc={val_acc:.4f}, f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained("best_model")

    # save file ───────────────────────────────────────────
    model = build_model().from_pretrained("best_model").to(device).eval()
    preds = []
    for batch in test_dl:
        ids = batch.pop("input_ids")
        mask = batch.pop("attention_mask")
        logits = model(ids.to(device), attention_mask=mask.to(device)).logits
        preds.extend(logits.argmax(-1).cpu().tolist())

    # Kaggle CSV
    import pandas as pd
    pd.DataFrame({"id": list(range(len(preds))), "label": preds}) \
      .to_csv("submission.csv", index=False)
    print("Saved → submission.csv")


if __name__ == "__main__":
    main(None)
