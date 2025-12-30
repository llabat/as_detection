import os
import sys
import yaml
import wandb
import optuna
from pathlib import Path
import torch

from trainer import train_partition_segmenter   # your existing trainer
from data_loader import build_dataset_from_gold
from transformers import AutoTokenizer

# Force wandb offline if WANDB_MODE=offline in SLURM env
mode = os.environ.get("WANDB_MODE", "online")

# -----------------------------------------------------
# Load config
# -----------------------------------------------------
def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# -----------------------------------------------------
# Objective function for Optuna
# -----------------------------------------------------
def objective(trial, as_dataset, config, model_name, type2id):

    # -----------------------------
    # 1. Sample hyperparameters
    # -----------------------------
    lr = trial.suggest_loguniform("lr", 3e-5, 1e-4)
    dropout = trial.suggest_float("dropout", 0.05, 0.25)
    type_loss_weight = trial.suggest_float("type_loss_weight", 0.8, 1.4)
    max_span_len = trial.suggest_int("max_span_len", 200, 280)
    # -----------------------------
    # 2. Start WandB trial run
    # -----------------------------
    wandb_run = wandb.init(
        project="as-segmenter-hp-tuning",
        mode=mode,   # <<<< crucial
        config={
            "trial": trial.number,
            "lr": lr,
            "dropout": dropout,
            "type_loss_weight": type_loss_weight,
            "max_span_len": max_span_len,
            "num_epochs": config["num_epochs"],
            "batch_size": config["batch_size"],
            "model_name": model_name,
        },
        reinit=True
    )

    # -----------------------------
    # 3. Train model
    # -----------------------------
    model, _ = train_partition_segmenter(
        model_name=model_name,
        train_dataset=as_dataset["train"],
        valid_dataset=as_dataset["eval"],
        type2id=type2id,
        out_dir=f"{config['model_output_dir']}/trial_{trial.number}",
        num_epochs=config["num_epochs"],
        batch_size=config["batch_size"],
        lr=lr,
        max_span_len=max_span_len,
        type_loss_weight=type_loss_weight,
        dropout=dropout,
        trial = trial,
        wandb_run = wandb_run
    )

    # -----------------------------
    # 4. Report to WandB & Optuna
    # -----------------------------
    best_f1 = model.best_val_f1
    wandb_run.log({"best_val_f1": best_f1})

    wandb_run.finish()

    # Pruning (optional)
    trial.report(best_f1, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return best_f1


# -----------------------------------------------------
#  MAIN: Setup sweep + run study
# -----------------------------------------------------

if __name__ == "__main__":

    # Load config
    config_path = Path(sys.argv[1])
    config = load_config(config_path)

    model_name = config["model_name"]
    type2id = {"premise": 0, "solution": 1, "claim": 2}

    # ----------------------------------------
    # Load dataset ONCE (super important)
    # ----------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    as_dataset = build_dataset_from_gold(config["path2json"], tokenizer, type2id)

    # ----------------------------------------
    # Define Optuna study storage (resumable)
    # ----------------------------------------
    study_path = f"{config['model_output_dir']}/optuna_study.db"
    pruner = optuna.pruners.MedianPruner(
    n_startup_trials=10,     # do not prune first 10 trials
    n_warmup_steps=1,        # allow first epoch to run
)

    study = optuna.create_study(
    direction="maximize",
    study_name="as_span_segmentation",
    storage=f"sqlite:///{study_path}",
    load_if_exists=True,
    pruner=pruner,
)

    # ----------------------------------------
    # Optimize
    # ----------------------------------------
    print("🚀 Starting Optuna hyperparameter sweep...")
    study.optimize(
        lambda trial: objective(trial, as_dataset, config, model_name, type2id),
        n_trials=config.get("n_trials", 20),   # configurable number of trials
        gc_after_trial=True,
    )

    # ----------------------------------------
    # Print results
    # ----------------------------------------
    print("🏆 Best Trial:")
    print(study.best_trial)

    # Save results
    study_file = f"{config['model_output_dir']}/best_trial_params.json"
    with open(study_file, "w") as f:
        import json
        json.dump(study.best_trial.params, f, indent=2)

    print(f"Saved best trial params to {study_file}")
