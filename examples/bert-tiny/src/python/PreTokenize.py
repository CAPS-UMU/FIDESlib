#!/usr/bin/env python3
"""
Pre-tokenize SST-2 validation sentences for cluster use (no Python at runtime).

Usage:
  python3 PreTokenize.py [--num_samples N]

This script:
  1. Loads SST-2 validation set
  2. Extracts BERT embeddings using prajjwal1/bert-tiny + SST-2-BERT-tiny.bin weights
  3. Saves embeddings to weights/weights-bert-tiny-sst2/pretokenized/sample_XXXX.txt
  4. Creates manifest.txt with sample metadata
"""

import sys
import os
import argparse
import torch
from transformers import BertForSequenceClassification, BertTokenizer, logging
from datasets import load_dataset

# Suppress warnings
logging.set_verbosity_error()

# Fixed configuration - same as save_weights.py
MODEL_ID = "prajjwal1/bert-tiny"
WEIGHTS_FILE = "SST-2-BERT-tiny.bin"
OUTPUT_DIR = "../../weights/weights-bert-tiny-sst2/pretokenized"


def extract_embeddings(model, tokenizer, text: str) -> torch.Tensor:
    """Extract BERT embeddings for a single sentence."""
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True, truncation=True)
    input_ids = enc["input_ids"]
    token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids))
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)
    
    with torch.no_grad():
        x = model.bert.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        )[0]
    return x


def save_embeddings(embeddings: torch.Tensor, filepath: str):
    """Save embeddings to file (one row per token, space-separated floats)."""
    with open(filepath, "w") as f:
        for row in embeddings:
            f.write(" ".join(f"{v.item():.12f}" for v in row) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize SST-2 for cluster use")
    parser.add_argument("--num_samples", "-n", type=int, default=100, 
                        help="Number of samples to pre-tokenize (default: 100)")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum token length (skip longer sequences)")
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, OUTPUT_DIR)
    weights_path = os.path.join(script_dir, WEIGHTS_FILE)
    
    print(f"Model: {MODEL_ID}")
    print(f"Weights: {weights_path}")
    print(f"Output: {out_dir}")
    
    # Load tokenizer and model - same setup as save_weights.py
    tokenizer = BertTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    model = BertForSequenceClassification.from_pretrained(MODEL_ID)
    
    # Load fine-tuned weights
    if os.path.isfile(weights_path):
        trained = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(trained, strict=False)
        print(f"Loaded fine-tuned weights from {weights_path}")
    else:
        print(f"WARNING: Fine-tuned weights not found at {weights_path}")
    
    model.eval()
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Load SST-2 validation set
    print("Loading SST-2 validation dataset...")
    dataset = load_dataset("glue", "sst2", split="validation")
    
    manifest_entries = []
    processed = 0
    skipped = 0
    
    print(f"Processing up to {args.num_samples} samples...")
    
    for idx, sample in enumerate(dataset):
        if processed >= args.num_samples:
            break
        
        try:
            sentence = sample["sentence"]
            label = sample["label"]
            sample_idx = sample.get("idx", idx)
            
            # Extract embeddings
            embeddings = extract_embeddings(model, tokenizer, sentence)
            token_length = embeddings.shape[0]
            
            if token_length > args.max_length:
                print(f"  Skipping sample {sample_idx}: token_length={token_length} > {args.max_length}")
                skipped += 1
                continue
            
            # Save embeddings
            filename = f"sample_{processed:04d}.txt"
            filepath = os.path.join(out_dir, filename)
            save_embeddings(embeddings, filepath)
            
            # Clean text for manifest
            clean_text = sentence.replace("\n", " ").replace("\t", " ").replace(",", " ")
            
            manifest_entries.append({
                "idx": sample_idx,
                "label": label,
                "token_length": token_length,
                "filename": filename,
                "text": clean_text
            })
            
            processed += 1
            if processed % 10 == 0:
                print(f"  Processed {processed}/{args.num_samples} samples...")
        
        except Exception as e:
            print(f"  Error processing sample {idx}: {e}")
            skipped += 1
            continue
    
    # Write manifest
    manifest_path = os.path.join(out_dir, "manifest.txt")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("# idx,label,token_length,filename,text\n")
        for entry in manifest_entries:
            f.write(f"{entry['idx']},{entry['label']},{entry['token_length']},{entry['filename']},{entry['text']}\n")
    
    print(f"\nDone! Processed {processed} samples, skipped {skipped}")
    print(f"Embeddings saved to: {out_dir}/")
    print(f"Manifest saved to: {manifest_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
