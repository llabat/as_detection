# evaluate_model.py

import os
import jsonlines
from typing import List, Dict, Any
from tqdm.auto import tqdm
import torch

from evaluation_script import span_evaluation   # <-- adjust path if needed


def decode_predictions(model, batch, id2type, max_span_len):
    """
    Runs the model in inference mode on a batch and returns a structured list of span predictions.
    """

    with torch.no_grad():
        logits_start, logits_end, logits_type = model.predict(
            batch["input_ids"],
            batch["attention_mask"],
            max_span_len=max_span_len
        )

    # Convert logits to start/end predictions
    start_probs = logits_start.sigmoid()
    end_probs   = logits_end.sigmoid()
    type_probs  = logits_type.softmax(-1)

    batch_predictions = []

    # Loop over batch elements
    for b in range(len(batch["text"])):
        text = batch["text"][b]
        doc_id = batch["doc_id"][b]

        # get predicted start/end tokens
        pred_starts = (start_probs[b] > 0.5).nonzero().flatten().tolist()
        pred_ends   = (end_probs[b] > 0.5).nonzero().flatten().tolist()

        spans = []
        for s in pred_starts:
            # find the nearest end >= s
            ends_after_s = [e for e in pred_ends if e >= s]
            if not ends_after_s:
                continue
            e = min(ends_after_s)

            # get type
            type_id = type_probs[b, s].argmax().item()
            type_str = id2type[type_id]

            # convert tokens back to text span
            tok_offsets = batch["offset_mapping"][b]
            char_start = tok_offsets[s][0]
            char_end   = tok_offsets[e][1]
            span_text  = text[char_start:char_end]

            spans.append({
                "text": span_text,
                "type": type_str
            })

        batch_predictions.append({
            "id": doc_id,
            "parsed_spans": spans,
            "text": text
        })

    return batch_predictions



def evaluate_model_on_test(
    model,
    tokenizer,
    test_loader,
    id2type,
    save_jsonl: str | None = None,
    overlap_threshold: float = 0.5,
    similarity: str = "min_ratio"
):
    """
    Full evaluation pipeline:
    - Runs inference over test set
    - Formats predictions to match evaluation script
    - Optionally saves JSONL predictions
    - Calls span_evaluation() and returns metrics
    """

    print("🚀 Running inference...")
    all_predictions = []

    model.eval()
    device = next(model.parameters()).device

    for batch in tqdm(test_loader, desc="Inference"):
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        preds = decode_predictions(model, batch, id2type, model.max_span_len)
        all_predictions.extend(preds)

    # ---------------------------
    # SAVE PREDICTIONS IF REQUESTED
    # ---------------------------
    if save_jsonl is not None:
        os.makedirs(os.path.dirname(save_jsonl), exist_ok=True)
        with jsonlines.open(save_jsonl, "w") as writer:
            for p in all_predictions:
                writer.write(p)
        print(f"💾 Saved predictions to {save_jsonl}")

    # ---------------------------
    # FORMAT FOR EVALUATION SCRIPT
    # ---------------------------
    pred_map = {p["id"]: [{"parsed_spans": p["parsed_spans"]}] for p in all_predictions}

    # User must load gold data from disk (same structure as evaluation_script)
    # but we expect:
    # gold_data[id] = list of gold annotations (list of entries per doc_id)

    print("⚖️ Evaluating...")

    # Instead of loading gold internally, we make user pass it in.
    # cleaner interface:
    def run_eval(gold_data):
        return span_evaluation(
            gold_data=gold_data,
            pred_data=pred_map,
            overlap_threshold=overlap_threshold,
            task="AS_EXTRACTION",
            similarity=similarity,
            print_results=True
        )

    return run_eval
