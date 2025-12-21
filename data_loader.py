import re
import json
import json
import ast
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding

class SegmentPreprocessor:
    
    def __init__(self, tokenizer, type2id, max_length=512):
        self.tokenizer = tokenizer
        self.type2id = type2id
        self.max_length = max_length

    def __call__(self, example):
        text = example["text"]
        spans = example["segments"]
    
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
        )
    
        offsets = encoding["offset_mapping"]
        # Identify which tokens are actual text (ignore CLS, SEP, Padding)
        # sequence_ids() returns None for special tokens
        sequence_ids = encoding.sequence_ids() 
    
        start_labels = [0] * len(offsets)
        end_labels   = [0] * len(offsets)
        span_starts, span_ends, span_types = [], [], []
    
        for seg in spans:
            ch_start, ch_end = seg["start"], seg["end"]
            ttype = self.type2id[seg["type"]]
    
            tok_start, tok_end = None, None
    
            for i, (s, e) in enumerate(offsets):
                # Skip special tokens (index 0 and index -1 usually)
                if sequence_ids[i] is None:
                    continue
                
                # Use 'closest match' logic
                # Start: first token that ends AFTER the character start
                if tok_start is None and e > ch_start:
                    tok_start = i
                
                # End: last token that starts BEFORE the character end
                if s < ch_end:
                    tok_end = i
    
            # Validation: Ensure we found tokens AND they are in the correct order
            if tok_start is not None and tok_end is not None and tok_start <= tok_end:
                start_labels[tok_start] = 1
                end_labels[tok_end] = 1
                span_starts.append(tok_start)
                span_ends.append(tok_end)
                span_types.append(ttype)
    
        encoding["start_labels"] = start_labels
        encoding["end_labels"] = end_labels
        encoding["span_starts"] = span_starts
        encoding["span_ends"] = span_ends
        encoding["span_types"] = span_types
        
        return encoding
        

class SpanCollator:
    def __init__(self, tokenizer):
        self.pad = DataCollatorWithPadding(tokenizer)

    def __call__(self, batch):
        span_starts = [ex["span_starts"] for ex in batch]
        span_ends   = [ex["span_ends"] for ex in batch]
        span_types  = [ex["span_types"] for ex in batch]

        keep = ("input_ids", "attention_mask", "token_type_ids")
        batch_for_pad = [{k: ex[k] for k in keep if k in ex} for ex in batch]

        padded = self.pad(batch_for_pad)
        padded["span_starts"] = span_starts
        padded["span_ends"]   = span_ends
        padded["span_types"]  = span_types
        return padded
        
        
def parse_as_json(path2json):
    with open(path2json, "r") as f:
        dicts = [json.loads(l) for l in f]
        
    as_dataset = []
    for dic in dicts:
        as_data = {"text" : dic["au_text"],
                   "split" : dic["split"]}
        
        for seg in dic["spans_types"]:
            start = dic["au_text"].find(seg["text"])
            end = start + len(seg["text"])
            as_data["segments"] = as_data.get("segments", []) + [{"start" : start, "end" : end, "type" : seg["type"]}]
            assert seg["text"] == dic["au_text"][start: end]
    
        as_dataset.append(as_data)
        
    return as_dataset
    

def build_dataset_from_gold(path2json, tokenizer, type2id, max_length=512):

    data = parse_as_json(path2json)
    dataset = Dataset.from_list(data)

    preproc = SegmentPreprocessor(tokenizer, type2id, max_length=max_length)

    encoded = dataset.map(
        preproc,
        batched=False,
        remove_columns=["text", "segments"]
    )

    # Build DatasetDict from the existing split column
    dataset_dict = DatasetDict({
        "train": encoded.filter(lambda x: x["split"] == "train"),
        "eval":  encoded.filter(lambda x: x["split"] == "eval"),
        "test":  encoded.filter(lambda x: x["split"] == "test"),
    })

    # Optionally drop the split column afterwards
    dataset_dict = dataset_dict.remove_columns(["split"])

    return dataset_dict
    

def check_mapping_efficiency(path2json, processed_dataset_dict, tokenizer, max_length=512):
    import json
    
    # 1. Get raw counts from the source file
    raw_data = []
    with open(path2json, "r") as f:
        for line in f:
            raw_data.append(json.loads(line))
            
    # Map raw data by text (or index) to find them in the processed dataset
    # We'll assume the order is preserved through Dataset.from_list
    
    total_gold_spans = 0
    total_mapped_spans = 0
    total_examples = 0
    empty_examples = 0

    # Combine splits for a full check
    all_processed = []
    for split in processed_dataset_dict.keys():
        for ex in processed_dataset_dict[split]:
            all_processed.append(ex)

    for i, raw_ex in enumerate(raw_data):
        gold_spans_in_this_ex = len(raw_ex["spans_types"])
        total_gold_spans += gold_spans_in_this_ex
        
        # Corresponding processed example
        proc_ex = all_processed[i]
        mapped_spans_in_this_ex = len(proc_ex["span_starts"])
        total_mapped_spans += mapped_spans_in_this_ex
        
        if mapped_spans_in_this_ex == 0:
            empty_examples += 1
            
        total_examples += 1

    print(f"--- Dataset Mapping Report ---")
    print(f"Total Examples: {total_examples}")
    print(f"Total Gold Spans in JSON: {total_gold_spans}")
    print(f"Total Spans Mapped to Tokens: {total_mapped_spans}")
    
    loss = total_gold_spans - total_mapped_spans
    if loss > 0:
        print(f"⚠️ Warning: Lost {loss} spans ({100*loss/total_gold_spans:.2f}%) during preprocessing.")
        print(f"🛑 {empty_examples} examples now have ZERO valid segments.")
    else:
        print("✅ 100% of spans were successfully mapped.")

    # Usage:
    # check_mapping_efficiency(path2json, as_dataset, tokenizer)