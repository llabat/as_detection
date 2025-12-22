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


def train_partition_segmenter(
    model_name: str,
    train_dataset,
    valid_dataset,
    type2id: dict,
    out_dir: str = "checkpoints_partition",
    num_epochs: int = 3,
    batch_size: int = 4,
    lr: float = 5e-5,
    max_span_len: int = 250,
    type_loss_weight: float = 1.0,
    proj_dim: int = 768,
    dropout: float = 0.1,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PartitionSpanSegmenter(
        model_name=model_name,
        num_types=len(type2id),
        max_span_len=max_span_len,
        proj_dim=proj_dim,
        dropout=dropout
    ).to(device)

    collator = SpanCollator(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    max_train_steps = num_epochs * len(train_loader)
    warmup_steps = int(0.1 * max_train_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    id2type = {v: k for k, v in type2id.items()}
    best_path = os.path.join(out_dir, "best.pt")

    # ----------------------------
    # For tracking training loss
    # ----------------------------
    running_loss = 0.0

    for epoch in range(1, num_epochs + 1):
        # ---------------- TRAIN ----------------
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=True)

        batch_count = 0
        running_loss = 0.0

        for batch in progress:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            span_starts    = batch["span_starts"]
            span_ends      = batch["span_ends"]
            span_types     = batch["span_types"]

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                span_starts=span_starts,
                span_ends=span_ends,
                span_types=span_types,
                type_loss_weight=type_loss_weight,
            )

            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Track average training loss
            running_loss += loss.item()
            batch_count += 1

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                end=f"{outputs['loss_end'].item():.4f}",
                type=f"{outputs['loss_type'].item():.4f}",
            )

        avg_train_loss = running_loss / batch_count

        # ---------------- VALID ----------------
        model.eval()
        val_f1 = evaluate_f1(model, valid_loader, tokenizer, type2id, id2type, max_span_len)
        print(f"Epoch {epoch}: validation micro-F1 = {val_f1:.4f}")

        # ---------------- W&B logging ----------------
        if wandb.run is not None:    # only if inside wandb.init()
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_f1": val_f1,
            })

        # ---------------- Save best ----------------
        if val_f1 > model.best_val_f1:
            model.best_val_f1 = val_f1
            torch.save(model.state_dict(), best_path)
            print("  ✓ saved new best model")

    return model, tokenizer