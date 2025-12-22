import sys
import yaml
import json
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from data_loader import build_dataset_from_gold
from trainer import train_partition_segmenter

def load_config(config_path):
    """Loads configuration from a YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

if __name__ == "__main__":

    config_path = Path(sys.argv[1])
    if not config_path.exists():
        raise FileNotFoundError(f"The provided config path does not exist: {path}")
        
    config = load_config(config_path)

    model_name = config["model_name"]
    TYPE2ID = {"premise": 0, "solution": 1, "claim": 2}
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    as_dataset = build_dataset_from_gold(config["path2json"], tokenizer, TYPE2ID)

    model, tokenizer = train_partition_segmenter(
        model_name = model_name,
        train_dataset = as_dataset["train"],
        valid_dataset = as_dataset["eval"],
        type2id = TYPE2ID,
        out_dir = config["model_output_dir"],
        num_epochs = config["num_epochs"],
        batch_size = config["batch_size"],
        lr = float(config["lr"]),
        max_span_len = config["max_span_len"],
        type_loss_weight = config["type_loss_weight"],
        proj_dim = config["proj_dim"],
        dropout = config["dropout"],
    )