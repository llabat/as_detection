import torch

import sys
from pathlib import Path
# Adjust this depending on where the repo is relative to your script
REPO_ROOT = Path("/home/labat/granddebat/gdnannotanalysis-main")
sys.path.append(str(REPO_ROOT))

from misc.span_evaluation import calculate_best_match_f1

# ---------------------------------------------------------
# 1. GREEDY DECODING (token-level)
# ---------------------------------------------------------

def greedy_decode(model, input_ids, attention_mask, tokenizer, max_span_len):
    """
    Predict spans greedily:
        returns list of (start_tok, end_tok, predicted_type)
    """
    model.eval()

    # Ensure device consistency
    device = next(model.parameters()).device
    input_ids_ = input_ids.unsqueeze(0).to(device)
    mask_ = attention_mask.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model.encoder(input_ids=input_ids_, attention_mask=mask_)
        h = out.last_hidden_state[0]       # (T, H)
        cls = h[0]                         # (H,)

    T = attention_mask.sum().item()
    s = 1  # skip CLS token
    pred_spans = []

    while s < T - 1:
        e_max = min(s + max_span_len - 1, T - 2)
        cand_indices = list(range(s, e_max + 1))

        # Build candidate span representations
        span_reprs = torch.stack(
            [model._span_repr(h, cls, s, e) for e in cand_indices],
            dim=0
        )

        scores, type_logits = model.span_head(span_reprs)
        best_idx = torch.argmax(scores).item()

        e_pred = cand_indices[best_idx]
        t_pred = torch.argmax(type_logits[best_idx]).item()

        pred_spans.append((s, e_pred, t_pred))

        s = e_pred + 1  # greedy jump

    return pred_spans



# ---------------------------------------------------------
# 2. TOKEN SPAN → RAW TEXT SPAN (predictions)
# ---------------------------------------------------------

def extract_text_spans(text, tokenizer, pred_spans):
    """
    Convert predicted (tok_start, tok_end, type) into:
        list of predicted span texts
        list of predicted span types
    """
    encoding = tokenizer(text, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]

    pred_texts = []
    pred_types = []
    
    for s, e, t in pred_spans:
        char_start = offsets[s][0]
        char_end   = offsets[e][1]
        span_text  = text[char_start:char_end]

        pred_texts.append(span_text)
        pred_types.append(t)
    
    return pred_texts, pred_types



# ---------------------------------------------------------
# 3. GOLD TOKEN SPANS → RAW TEXT SPANS
# ---------------------------------------------------------

def build_gold_text_spans(text, tokenizer, starts, ends, types, id2type):
    """
    Convert gold token boundaries into raw text spans.
    """
    encoding = tokenizer(text, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]

    spans = []
    span_types = []

    for s, e, t in zip(starts, ends, types):
        char_start = offsets[s][0]
        char_end   = offsets[e][1]
        span_text  = text[char_start:char_end]

        spans.append(span_text)
        span_types.append(id2type[t])

    return spans, span_types



# ---------------------------------------------------------
# 4. VALIDATION F1 (MICRO)
# ---------------------------------------------------------

def evaluate_f1(model, valid_loader, tokenizer, type2id, id2type, max_span_len):
    """
    Computes micro-F1 over the entire validation loader.
    """

    total_TP = 0
    total_FP = 0
    total_FN = 0

    model.eval()

    for batch in valid_loader:
        input_ids      = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        texts          = batch["text"]
        gold_starts    = batch["span_starts"]
        gold_ends      = batch["span_ends"]
        gold_types     = batch["span_types"]

        B = input_ids.size(0)

        for i in range(B):
            ids = input_ids[i]
            mask = attention_mask[i]
            text = texts[i]

            # ------------------ 1. Predict spans ------------------
            pred_tok_spans = greedy_decode(
                model, ids, mask, tokenizer, max_span_len
            )

            # ------------------ 2. Predicted raw-text spans ------------------
            pred_spans_text, pred_types = extract_text_spans(
                text, tokenizer, pred_tok_spans
            )
            pred_type_labels = [id2type[t] for t in pred_types]

            # ------------------ 3. Gold raw-text spans ------------------
            gold_text_spans, gold_type_labels = build_gold_text_spans(
                text,
                tokenizer,
                gold_starts[i],
                gold_ends[i],
                gold_types[i],
                id2type
            )

            # ------------------ 4. Compute TP / FP / FN ------------------
            _, _, _, TP, FP, FN = calculate_best_match_f1(
                gold_text_spans, 
                pred_spans_text, 
                gold_type_labels, 
                pred_type_labels,
                overlap_threshold=0.5,
                similarity="min_ratio"
            )

            total_TP += TP
            total_FP += FP
            total_FN += FN

    # ---------------------- MICRO F1 ----------------------
    P = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    R = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    F1 = (2 * P * R) / (P + R) if (P + R) > 0 else 0.0

    return F1