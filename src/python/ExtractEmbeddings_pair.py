# /projectnb/he/seyda/FIDESlib/src/python/ExtractEmbeddings_pair.py
import sys, os, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging
logging.set_verbosity_error()

if len(sys.argv) != 6:
    print("Usage: python3 ExtractEmbeddings_pair.py <s1> <s2> <model_name> <out_dir> <out_fname>")
    sys.exit(2)

s1, s2, model_name, out_dir, out_fname = sys.argv[1:6]

# Allow either your alias or a direct HF id
model_id = "M-FAC/bert-tiny-finetuned-mrpc" if model_name == "bert-tiny-mrpc" else model_name

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
model.eval()

# Pair tokenization (segment ids matter for MRPC)
enc = tokenizer(s1, s2, return_tensors="pt", add_special_tokens=True, truncation=True)
input_ids = enc["input_ids"]                                  # [1, seq]
token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids))
position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)

with torch.no_grad():
    # [1, seq, hidden] -> [seq, hidden]
    x = model.bert.embeddings(input_ids=input_ids,
                              token_type_ids=token_type_ids,
                              position_ids=position_ids)[0]

os.makedirs(out_dir, exist_ok=True)
path = os.path.join(out_dir, out_fname)
with open(path, "w") as f:
    for row in x:
        # 18-digit scientific notation to match your HF-style dump
        f.write(" ".join(f"{v.item():.18e}" for v in row) + "\n")

# C++ side reads this to get seq_len
print(x.shape[0])
