# /projectnb/he/seyda/FIDESlib/src/python/ExtractEmbeddings_pair.py
import sys, os, torch
from transformers import BertForSequenceClassification, BertTokenizer, logging
logging.set_verbosity_error()

USAGE = "Usage: python3 ExtractEmbeddings_pair.py <s1> <s2> <model_name> <out_dir> <out_fname>"

if len(sys.argv) != 6:
    print(USAGE)
    sys.exit(2)

s1, s2, model_name, out_dir, out_fname = sys.argv[1:6]

# --- hard-coded aliases ---
ALIASES = {
    # MRPC
    "mrpc": "M-FAC/bert-tiny-finetuned-mrpc",
    "bert-tiny-mrpc": "M-FAC/bert-tiny-finetuned-mrpc",
    # RTE
    "rte": "muhtasham/bert-tiny-finetuned-glue-rte",
    "bert-tiny-rte": "muhtasham/bert-tiny-finetuned-glue-rte",
}

def resolve_model_id(name: str) -> str:
    # If it's a local directory with saved tokenizer/model, use it directly.
    if os.path.isdir(name):
        return name
    # Otherwise map known aliases or pass through as an HF repo id.
    return ALIASES.get(name, name)

model_id = resolve_model_id(model_name)

# Optional auth/offline controls
hf_token = os.getenv("HF_TOKEN", None)
local_only = bool(os.getenv("HF_OFFLINE"))

tokenizer = BertTokenizer.from_pretrained(
    model_id,
    use_fast=True,
    local_files_only=local_only)

model = BertForSequenceClassification.from_pretrained(
    model_id,
    local_files_only=local_only)
model.eval()

# Pair tokenization (segment ids matter for pair tasks)
enc = tokenizer(
    s1, s2,
    return_tensors="pt",
    add_special_tokens=True,
    truncation=True,
)
input_ids = enc["input_ids"]                              # [1, seq]
token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids))
position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)

with torch.no_grad():
    # [1, seq, hidden] -> [seq, hidden]
    x = model.bert.embeddings(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        position_ids=position_ids
    )[0]

os.makedirs(out_dir, exist_ok=True)
path = os.path.join(out_dir, out_fname)
with open(path, "w") as f:
    for row in x:
        f.write(" ".join(f"{v.item():.18e}" for v in row) + "\n")

# C++ side reads this stdout to get seq_len
print(x.shape[0])


