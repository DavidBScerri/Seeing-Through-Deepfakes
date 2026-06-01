import os
import random
import collections
from datasets import load_dataset
from huggingface_hub import list_repo_files

files = list_repo_files("Zitacron/real-vs-ai-corpus", repo_type="dataset")
parquet_files = sorted([f for f in files if f.endswith(".parquet")])
random.seed(42)
random.shuffle(parquet_files)

collected = collections.defaultdict(list)
BUFFER_SIZE = 50

for file in parquet_files[:5]: # just test first 5 files
    print(f"Reading {file}...")
    ds = load_dataset("Zitacron/real-vs-ai-corpus", data_files=file, split="train", streaming=True)
    skipped = 0
    for row in ds:
        src = row["source_dataset"]
        if len(collected[src]) < BUFFER_SIZE:
            collected[src].append(row["label"])
            skipped = 0
            if len(collected[src]) % 10 == 0:
                print(f"  {src}: {len(collected[src])}")
        else:
            skipped += 1
            
        if skipped > 50:
            print(f"  Skipped {skipped} rows, breaking file.")
            break
            
print("Sources found:", collected.keys())
