"""
Prepara TinyStories → train.bin / val.bin (uint16).
Normalização Unicode completa + sanitização ASCII.
"""

import os
import re
import pickle
import unicodedata
import numpy as np
from collections import Counter
from datasets import load_dataset
from tqdm.auto import tqdm

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_NAME = "roneneldan/TinyStories"
VOCAB_SIZE = 8192
CHUNK = 100_000

TOKEN_RE = re.compile(r"[a-z0-9]+|[.,!?;:'\"-]")


def normalize(text):
    text = unicodedata.normalize('NFKC', text)
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    text = text.replace('—', '-').replace('–', '-')
    text = text.replace('…', '...')
    text = text.replace('\u00a0', ' ')
    text = text.lower()
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text


def tokenize(text):
    return TOKEN_RE.findall(normalize(text))


def main():
    print(f"Baixando {DATASET_NAME} do Hugging Face...")
    ds = load_dataset(DATASET_NAME)
    print(ds)

    print("\n[1/3] Contando palavras (train)...")
    counter = Counter()
    n = len(ds["train"])
    for i in tqdm(range(0, n, CHUNK), desc="contando"):
        batch = ds["train"][i:i + CHUNK]["text"]
        for t in batch:
            counter.update(tokenize(t))

    most_common = counter.most_common(VOCAB_SIZE - 2)
    itos = ['<pad>', '<unk>'] + [w for w, _ in most_common]
    stoi = {w: i for i, w in enumerate(itos)}
    total = sum(counter.values())
    cobertura = sum(c for _, c in most_common) / total * 100
    print(f"vocab: {len(itos):,} | cobertura: {cobertura:.2f}%")

    print("\n[2/3] Convertendo train...")
    out_path = os.path.join(OUT_DIR, 'train.bin')
    total_ids = 0
    with open(out_path, 'wb') as f:
        for i in tqdm(range(0, n, CHUNK), desc="train"):
            batch = ds["train"][i:i + CHUNK]["text"]
            ids = []
            for t in batch:
                ids.extend(stoi.get(w, 1) for w in tokenize(t))
            arr = np.array(ids, dtype=np.uint16)
            arr.tofile(f)
            total_ids += len(arr)
    print(f"train: {total_ids:,} IDs")

    print("\n[3/3] Convertendo val...")
    n_val = len(ds["validation"])
    out_path = os.path.join(OUT_DIR, 'val.bin')
    total_val = 0
    with open(out_path, 'wb') as f:
        for i in tqdm(range(0, n_val, CHUNK), desc="val"):
            batch = ds["validation"][i:i + CHUNK]["text"]
            ids = []
            for t in batch:
                ids.extend(stoi.get(w, 1) for w in tokenize(t))
            arr = np.array(ids, dtype=np.uint16)
            arr.tofile(f)
            total_val += len(arr)
    print(f"val: {total_val:,} IDs")

    with open(os.path.join(OUT_DIR, 'meta.pkl'), 'wb') as f:
        pickle.dump({'vocab_size': len(itos), 'itos': itos, 'stoi': stoi}, f)

    print(f"\nSalvos em {OUT_DIR}/")


if __name__ == '__main__':
    main()
