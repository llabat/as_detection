import os, math
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture import PartitionSpanSegmenter
from data_loader import SpanCollator

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
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PartitionSpanSegmenter(
        model_name=model_name,
        num_types=len(type2id),
        max_span_len=max_span_len,
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

    best_val = float("inf")

    for epoch in range(1, num_epochs + 1):
        # ---------------- TRAIN ----------------
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=True)

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

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                end=f"{outputs['loss_end'].item():.4f}",
                type=f"{outputs['loss_type'].item():.4f}",
            )

        # ---------------- VALID ----------------
        model.eval()
        val_sum = 0.0
        val_n = 0

        with torch.no_grad():
            for batch in valid_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    span_starts=batch["span_starts"],
                    span_ends=batch["span_ends"],
                    span_types=batch["span_types"],
                    type_loss_weight=type_loss_weight,
                )

                val_sum += outputs["loss"].item()
                val_n += 1

        avg_val = val_sum / max(1, val_n)
        print(f"Epoch {epoch}/{num_epochs} - valid loss: {avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            best_path = os.path.join(out_dir, "best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"  ✅ saved new best to {best_path}")

    return model, tokenizer