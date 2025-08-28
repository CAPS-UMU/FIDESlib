from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import sys
import os

from transformers import logging
logging.set_verbosity_error()

# Read command line args
if len(sys.argv) < 4:
    print("Usage: python3 ExtractEmbeddings.py \"<sentence>\" <model_name> <model_path> <output_filename>")
    sys.exit(1)

text = "[CLS] " + sys.argv[1] + " [SEP]"
model_name = sys.argv[2]
output_path = sys.argv[3]
output_filename = sys.argv[4]

trained = torch.load(output_path + '/../../src/python/SST-2-BERT-tiny.bin', map_location=torch.device('cpu'))

tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny")

model.load_state_dict(trained, strict=False)
model.eval()

# Tokenize
tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
tokens_tensor = torch.tensor([indexed_tokens])

# x = model.bert.embeddings(tokens_tensor, torch.tensor([[1] * len(tokenized_text)]))[0]
x = model.bert.embeddings(tokens_tensor)[0]

# Write embeddings
output_file = os.path.join(output_path, output_filename)
os.makedirs(output_path, exist_ok=True)

with open(output_file, 'w') as f:
    for row in x:
        line = ' '.join(f"{v:.12f}" for v in row.tolist())
        f.write(line + '\n')

print(x.shape[0])
