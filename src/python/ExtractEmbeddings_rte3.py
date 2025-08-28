# ExtractEmbeddings.py
import sys, os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging
logging.set_verbosity_error()

if len(sys.argv) < 5:
    print("Usage: python3 ExtractEmbeddings.py <sentence> <model_name> <model_path> <output_filename>")
    sys.exit(1)

text         = sys.argv[1]
model_name   = sys.argv[2]  
output_path  = sys.argv[3]
output_fname = sys.argv[4]

model_id = "muhtasham/bert-tiny-finetuned-glue-rte"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

model.eval()

enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
input_ids = enc["input_ids"]

with torch.no_grad():
    x = model.bert.embeddings(input_ids)[0]  # shape: [seq_len, hidden]

os.makedirs(output_path, exist_ok=True)
out_file = os.path.join(output_path, output_fname)
with open(out_file, "w") as f:
    for row in x:
        f.write(" ".join(f"{v.item():.12f}" for v in row) + "\n")

print(x.shape[0])  # token length to be parsed by C++
