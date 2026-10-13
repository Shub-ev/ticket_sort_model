# Ticket Assignment Classifier

Fine-tune DistilBERT to route incident tickets to an **Assignment Group** using
the ticket's short description and description.

## Setup

Use Python 3.12 or newer, create and activate a virtual environment, then
install the pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install --upgrade torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

The PyTorch install above replaces the CPU-only wheel with the CUDA 12.8 build
matching the version pinned in `requirements.txt`. It is for NVIDIA GPUs and
requires a compatible NVIDIA driver. Confirm that PyTorch can see your GPU
before training:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA GPU not found')"
```

Put the source CSV at `data/raw/incidents.csv`. It must contain `Short
description`, `Description`, and `Assignment Group` columns. The data can
contain sensitive incident information; keep raw data, processed splits, and
trained artifacts private.

## Train and evaluate

Run the pipeline from the repository root:

```powershell
python data/validate_dataset.py
python data/split_data.py
python training/train.py
python main.py
```

`main.py` creates `data/processed/ticket_test_set.json` from the held-out
`test.csv` split if it does not exist, then loads the trained model and reports
accuracy and per-group metrics. Keep this file private because its ticket text
comes from the incident dataset.

The split step removes exact duplicate ticket texts after checking that a
ticket text does not have conflicting assignment-group labels. It then makes
reproducible, stratified 80/10/10 train/validation/test splits. Keeping exact
duplicates in different splits can make evaluation misleading.

Training requires CUDA by default so it will not silently start a slow CPU
run. For a machine with a small GPU, the default per-device batch size is 2
with gradient accumulation; reduce `--batch-size` to 1 if CUDA reports out of
memory. Use `--allow-cpu` only to explicitly opt into CPU training.

Training uses dynamically padded inputs, square-root class weights, early
stopping, and the validation macro-F1 score to choose the best checkpoint. The
test split is evaluated only after checkpoint selection. Optional training
settings can be changed with command-line arguments; run
`python training/train.py --help` for details.

The final model, tokenizer, `label_mapping.json`, per-class `evaluation.json`,
and `training_metadata.json` are saved to
`models/distilbert_ticket_classifier/`. A separate checkpoint history is kept
under `models/training_runs/`.

## Evaluation caveat

The current data has highly imbalanced assignment groups, including a group
with only seven distinct tickets. Macro-F1 and per-class test results are more
informative than accuracy alone, but metrics for groups with very few examples
will have high uncertainty. Collect more labeled examples for rare groups
before relying on automated routing for them. When reliable ticket timestamps
are available, also evaluate on a later time period to measure performance on
future tickets.
