import os, math
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture import PartitionSpanSegmenter
from data_loader import SpanCollator
from evaluation_span_model import evaluate_f1

import wandb   # W&B
import optuna

def train_partition_segmenter(
    model_name: str,
    train_dataset,
    valid_dataset,
    type2id: dict,
    out_dir: str = "checkpoints_partition",
    num_epochs: int = 10,
    patience: int = 2,
    min_delta: float = 1e-4,
    batch_size: int = 4,
    lr: float = 5e-5,
    max_span_len: int = 250,
    type_loss_weight: float = 1.0,
    dropout: float = 0.1,
    trial=None,          # NEW (Optuna)
    wandb_run=None       # NEW (WandB)
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PartitionSpanSegmenter(
        model_name=model_name,
        num_types=len(type2id),
        max_span_len=max_span_len,
        dropout=dropout,
    ).to(device)

    collator = SpanCollator(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    max_train_steps = num_epochs * len(train_loader)
    warmup_steps = int(0.1 * max_train_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        warmup_steps,
        max_train_steps,
    )

    id2type = {v: k for k, v in type2id.items()}

    best_f1 = -1.0
    best_path = os.path.join(out_dir, "best.pt")
    bad_epochs = 0

    for epoch in range(1, num_epochs + 1):

        # ---------------- TRAIN ----------------
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):

            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                span_starts=batch["span_starts"],
                span_ends=batch["span_ends"],
                span_types=batch["span_types"],
                type_loss_weight=type_loss_weight,
            )

            loss = outputs["loss"]
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

        epoch_loss /= len(train_loader)

        # ---------------- VALID ----------------
        model.eval()
        val_f1 = evaluate_f1(model, valid_loader, tokenizer, type2id, id2type, max_span_len)
        print(f"Epoch {epoch} → val F1 = {val_f1:.4f}")
        if trial is not None:
            trial.report(val_f1, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # ----- WandB logging -----
        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch,
                "train_loss": epoch_loss,
                "val_f1": val_f1,
                "lr": scheduler.get_last_lr()[0],
            })

        # ----- Optuna pruning -----
        if trial is not None:
            trial.report(val_f1, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # ----- Early stopping -----
        if val_f1 > best_f1 + min_delta:
            best_f1 = val_f1
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("⛔ Early stopping triggered")
                break

    model.best_val_f1 = best_f1
    return model, tokenizer

