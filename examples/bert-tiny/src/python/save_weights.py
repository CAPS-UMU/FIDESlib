from transformers import BertForSequenceClassification, BertTokenizer
import numpy as np
from datasets import load_dataset
import pandas as pd
import torch
import os

tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny", use_fast=False)
model = BertForSequenceClassification.from_pretrained("prajjwal1/bert-tiny")

trained = torch.load('SST-2-BERT-tiny.bin', map_location=torch.device('cpu'))
model.load_state_dict(trained , strict=False)
model.eval()

# Output directory
output_dir = "../../weights/weights-bert-tiny-sst2"
os.makedirs(output_dir, exist_ok=True)

# Load SST-2 dataset
dataset = load_dataset("glue", "sst2", split="validation")
df = dataset.to_pandas()
df.to_csv(output_dir + "/sst2_validation.csv", index=False)

def save(name, array):
    path = os.path.join(output_dir, f"{name}.txt")
    np.savetxt(path, array, fmt="%.12f")

save("tokens_sst2", np.zeros((4,4)))    # dummy file for tokens

# For layers 0 and 1
for i in range(2):
    prefix = f"layer{i}_"
    layer = model.bert.encoder.layer[i]

    # Attention weights (transposed)
    Wq = layer.attention.self.query.weight.detach().numpy().T
    Wk = layer.attention.self.key.weight.detach().numpy().T / 128.0 # Softmax scale
    Wv = layer.attention.self.value.weight.detach().numpy().T
    bq = layer.attention.self.query.bias.detach().numpy().reshape(-1, 1).T
    bk = layer.attention.self.key.bias.detach().numpy().reshape(-1, 1).T / 128.0 # Softmax scale
    bv = layer.attention.self.value.bias.detach().numpy().reshape(-1, 1).T

    save(prefix + "Wq", Wq)
    save(prefix + "Wk", Wk)
    save(prefix + "Wv", Wv)
    save(prefix + "bq", bq)
    save(prefix + "bk", bk)
    save(prefix + "bv", bv)

    # Attention output projection
    Wo = layer.attention.output.dense.weight.detach().numpy().T
    bo = layer.attention.output.dense.bias.detach().numpy().reshape(-1, 1).T
    save(prefix + "Wo", Wo)
    save(prefix + "bo", bo)

    # # Feed-forward layers (intermediate and output)
    Wff1 = layer.intermediate.dense.weight.detach().numpy().T / 20.0  # Chebyshev scale
    bff1 = layer.intermediate.dense.bias.detach().numpy().reshape(-1, 1).T / 20.0   # Chebyshev scale
    Wff2 = layer.output.dense.weight.detach().numpy().T *20
    bff2 = layer.output.dense.bias.detach().numpy().reshape(-1, 1).T

    save(prefix + "Wu", Wff1)
    save(prefix + "bu", bff1)
    save(prefix + "Wd", Wff2)
    save(prefix + "bd", bff2)

    Wln = layer.attention.output.LayerNorm.weight.detach().numpy().reshape(-1, 1).T
    bln = layer.attention.output.LayerNorm.bias.detach().numpy().reshape(-1, 1).T

    save(prefix + "Wln1", Wln)
    save(prefix + "bln1", bln)

    Wln = layer.output.LayerNorm.weight.detach().numpy().reshape(-1, 1).T
    bln = layer.output.LayerNorm.bias.detach().numpy().reshape(-1, 1).T

    save(prefix + "Wln2", Wln)
    save(prefix + "bln2", bln)

Wp = model.bert.pooler.dense.weight.detach().numpy().T
bp = model.bert.pooler.dense.bias.detach().numpy().reshape(-1, 1).T

save("Wp", Wp)
save("bp", bp)

Wc = model.classifier.weight.detach().numpy()
bc = model.classifier.bias.detach().numpy().reshape(-1, 1)

save("Wc", Wc)
save("bc", bc)