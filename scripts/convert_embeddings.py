#!/usr/bin/env python3
"""Convertit les CSV avec embeddings JSON-texte en .npy float32 + métadonnées parquet.

Entrées :
  data/objets_embed.csv                    (activités HATVP, 1536-d)
  data/amendments_2025_aligned_quoted.csv  (amendements 2025, 1536-d)

Sorties dans data/embeddings/ :
  hatvp_embeddings.npy / hatvp_meta.parquet
  amendments_embeddings.npy / amendments_meta.parquet
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "embeddings"
OUT.mkdir(parents=True, exist_ok=True)

DIM = 1536


def parse_vec(s: str) -> np.ndarray:
    return np.fromstring(s[1:-1], sep=",", dtype=np.float32)


def convert(csv_path: Path, out_prefix: str, embed_col: str) -> None:
    vecs = []
    meta_rows = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        # dédoublonne les noms de colonnes (jointure : date, dossier, ... apparaissent 2x)
        seen: dict[str, int] = {}
        cols = []
        for c in header:
            if c in seen:
                seen[c] += 1
                cols.append(f"{c}_{seen[c]}")
            else:
                seen[c] = 0
                cols.append(c)
        ei = header.index(embed_col)
        for row in reader:
            v = parse_vec(row[ei])
            if v.shape[0] != DIM:
                raise ValueError(f"{csv_path.name} ligne {len(vecs)+2}: dim {v.shape[0]}")
            vecs.append(v)
            meta_rows.append([x for i, x in enumerate(row) if i != ei])
    emb = np.vstack(vecs)
    meta_cols = [c for i, c in enumerate(cols) if i != ei]
    meta = pd.DataFrame(meta_rows, columns=meta_cols)
    np.save(OUT / f"{out_prefix}_embeddings.npy", emb)
    meta.to_parquet(OUT / f"{out_prefix}_meta.parquet")
    norms = np.linalg.norm(emb, axis=1)
    print(
        f"{out_prefix}: {emb.shape} float32, "
        f"normes min/med/max = {norms.min():.4f}/{np.median(norms):.4f}/{norms.max():.4f}",
        flush=True,
    )


if __name__ == "__main__":
    convert(ROOT / "data" / "amendments_2025_aligned_quoted.csv", "amendments", "embedding_objet")
    print("amendements OK", flush=True)
    convert(ROOT / "data" / "objets_embed.csv", "hatvp", "embedding_objet")
    print("HATVP OK", flush=True)
