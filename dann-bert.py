import json, random
from pathlib import Path
from typing import List, Dict
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from transformers import BertConfig, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

RANDOM_SEED = 42
MAX_LEN     = 256
BATCH_SIZE  = 16
EPOCHS      = 4
LR          = 2e-5
VOCAB_SIZE  = 17120
LAMBDA_ADV  = 0.1
DOMAIN_HID  = 256

torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

def read_json(path: Path, with_label: bool = True):
    data = []
    for obj in map(json.loads, path.read_text().splitlines()):
        ids = obj['text'][:MAX_LEN]
        pad = [0] * (MAX_LEN - len(ids))
        item = {
            'input_ids':      torch.tensor(ids + pad, dtype=torch.long),
            'attention_mask': torch.tensor([1]*len(ids) + pad, dtype=torch.long)
        }
        if with_label:
            item['labels'] = torch.tensor(obj['label'], dtype=torch.long)
        data.append(item)
    return data

class JsonDataset(Dataset):
    def __init__(self, samples: List[Dict]): self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# Gradient Reversal Layer
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg().mul(ctx.alpha), None

def grad_reverse(x, alpha):
    return GradientReversal.apply(x, alpha)

# Domain Discriminator
class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim, hid_dim=DOMAIN_HID):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hid_dim, 2)
        )
    def forward(self, x, alpha):
        x = grad_reverse(x, alpha)
        return self.net(x)

# model
class DANNModel(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = BertConfig(
            vocab_size=VOCAB_SIZE,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            max_position_embeddings=MAX_LEN,
            type_vocab_size=1,
            num_labels=2,
            output_hidden_states=True
        )
        self.bert = BertForSequenceClassification(cfg)
        self.domain_disc = DomainDiscriminator(cfg.hidden_size)
    def forward(self, input_ids, attention_mask, alpha=0.0):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        feat = out.hidden_states[-1][:,0,:]
        logits = out.logits
        dom_logits = self.domain_disc(feat, alpha)
        return feat, logits, dom_logits

def train_epoch(model, loader_s, loader_t, optimiser, scheduler, device, epoch):
    model.train()
    total, correct = 0, 0
    steps = min(len(loader_s), len(loader_t))
    desc = f"Train Epoch {epoch}/{EPOCHS}"

    for (bs, bt) in tqdm(zip(loader_s, loader_t), total=steps, desc=desc):
        labels_s = bs.pop('labels').to(device)
        bs = {k:v.to(device) for k,v in bs.items()}
        bt = {k:v.to(device) for k,v in bt.items()}

        p = float(epoch - 1) / EPOCHS
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # forward
        feat_s, logits_s, dom_s = model(**bs, alpha=alpha)
        _, _, dom_t = model(**bt, alpha=alpha)
        loss_cls = nn.CrossEntropyLoss()(logits_s, labels_s)
        ds = torch.zeros(dom_s.size(0), dtype=torch.long, device=device)
        dt = torch.ones (dom_t.size(0), dtype=torch.long, device=device)
        dom_logits = torch.cat([dom_s, dom_t], dim=0)
        dom_labels = torch.cat([ds, dt], dim=0)
        loss_dom = nn.CrossEntropyLoss()(dom_logits, dom_labels)

        loss = loss_cls + LAMBDA_ADV * loss_dom

        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        if scheduler: scheduler.step()

        preds = logits_s.argmax(-1)
        total += labels_s.size(0)
        correct += (preds == labels_s).sum().item()

    return correct / total

@torch.no_grad()
def eval_loop(model, loader, device):
    model.eval()
    ys, ps = [], []
    for batch in tqdm(loader, desc="Eval"):
        labels = batch.pop('labels').to(device)
        batch = {k:v.to(device) for k,v in batch.items()}
        logits = model(**batch)[1]
        ys.extend(labels.cpu().tolist())
        ps.extend(logits.argmax(-1).cpu().tolist())
    return accuracy_score(ys, ps), f1_score(ys, ps, average='macro')

def main(args):
    d1 = read_json(Path('domain1_train_data.json'), with_label=True)
    d2 = read_json(Path('domain2_train_data.json'), with_label=False)
    random.shuffle(d1); random.shuffle(d2)
    split1 = int(0.9*len(d1)); split2 = int(0.9*len(d2))
    ds_s_tr = JsonDataset(d1[:split1]); ds_t_tr = JsonDataset(d2[:split2])
    ds_val  = JsonDataset(d1[split1:])

    loader_s = DataLoader(ds_s_tr, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    loader_t = DataLoader(ds_t_tr, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE)
    test_ds = JsonDataset(read_json(Path('test_data.json'), with_label=False))
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DANNModel().to(device)
    optimiser = optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(loader_s) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimiser, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps
    )

    best_f1 = 0.0
    for ep in range(1, EPOCHS+1):
        tr_acc = train_epoch(model, loader_s, loader_t, optimiser, scheduler, device, ep)
        val_acc, val_f1 = eval_loop(model, loader_val, device)
        print(f"Epoch {ep}: train acc={tr_acc:.4f} | val acc={val_acc:.4f}, f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            model.bert.save_pretrained('best_dann_model')

    model.bert.from_pretrained('best_dann_model').to(device).eval()
    preds = []
    for batch in test_dl:
        batch = {k:v.to(device) for k,v in batch.items()}
        logits = model(**batch, alpha=0.0)[1]
        preds.extend(logits.argmax(-1).cpu().tolist())
    import pandas as pd
    pd.DataFrame({'id': list(range(len(preds))), 'label': preds}).to_csv('submission_dann.csv', index=False)
    print('Saved → submission_dann.csv')

if __name__ == '__main__':
    main(None)
