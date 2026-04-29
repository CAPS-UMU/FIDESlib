#!/usr/bin/env python3
# Usage: python3 ExtractEmbeddings.py <sentence> <dataset> <model_name> <out_dir> <out_fname> [mode]
import sys, os, torch
from transformers import BertForSequenceClassification, BertTokenizer, logging

# Suppress warnings
logging.set_verbosity_error()

USAGE = "Usage: python3 ExtractEmbeddings.py <sentence> <dataset> <model_name> <out_dir> <out_fname> [mode]"

# Allow 6 args (standard) or 7 args (including mode)
# argv[0] is the script name
if len(sys.argv) < 6 or len(sys.argv) > 7:
    print(USAGE)
    sys.exit(2)

# Parse standard arguments
text, dataset, model_name, out_dir, out_fname = sys.argv[1:6]
dataset = dataset.lower().strip()

# Parse optional mode argument (create vs update)
mode = "create"
if len(sys.argv) == 7:
    mode = sys.argv[6]

# Determine file open mode: 'w' for overwrite, 'a' for append
write_mode = 'w'

# ---- model resolution ----
def resolve_model_id(_dataset: str, _model_name: str) -> str:
    if os.path.isdir(_model_name):
        return _model_name
    defaults = {
        "sst2": "prajjwal1/bert-tiny",
        "cola": "Sayan01/tiny-bert-cola-128-distilled",
    }
    return _model_name if "/" in _model_name else defaults.get(_dataset, "prajjwal1/bert-tiny")

model_id = resolve_model_id(dataset, model_name)

# Optional offline/auth controls
hf_token = os.getenv("HF_TOKEN", None)
local_only = bool(os.getenv("HF_OFFLINE"))

tokenizer = BertTokenizer.from_pretrained(
    model_id, use_fast=True, local_files_only=local_only)
model = BertForSequenceClassification.from_pretrained(
    model_id, local_files_only=local_only)
model.eval()

# Load custom weights for SST-2 if available
if dataset == "sst2":
    cand = os.getenv("MODEL_BIN")
    if not cand:
        cand = os.path.abspath(os.path.join(out_dir, "..", "..", "src", "python", "SST-2-BERT-tiny.bin"))
    if os.path.isfile(cand):
        try:
            sd = torch.load(cand, map_location="cpu")
            model.load_state_dict(sd, strict=False)
        except Exception as e:
            print(f"[warn] could not load SST-2 finetuned weights from {cand}: {e}", file=sys.stderr)

# ---- tokenize ----
enc = tokenizer(text, return_tensors="pt", add_special_tokens=True, truncation=True)
input_ids = enc["input_ids"]
token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids))
position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)

# ---- extract embeddings ----
with torch.no_grad():
    x = model.bert.embeddings(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        position_ids=position_ids
    )[0]

# ---- write to file ----
os.makedirs(out_dir, exist_ok=True)
path = os.path.join(out_dir, out_fname)

# Open using the dynamic write_mode ('w' or 'a')
with open(path, write_mode) as f:
    for row in x:
        f.write(" ".join(f"{v.item():.12f}" for v in row) + "\n")

# Print seq length for the C++ caller
print(x.shape[0])