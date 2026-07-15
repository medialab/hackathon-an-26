#!/usr/bin/env python3
"""Convertit subset_facebook_posts_matched.csv (posts Facebook de lobbyistes,
embedding_text = vecteur 1536-d) en .npy + parquet, même convention que
convert_embeddings.py.

Entrée : subset_facebook_posts_matched.csv (racine du repo)
Sorties dans data/embeddings/ : facebook_embeddings.npy / facebook_meta.parquet
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "embeddings"
OUT.mkdir(parents=True, exist_ok=True)

DIM = 1536


def main() -> None:
    src = ROOT / "subset_facebook_posts_matched.csv"
    df = pd.read_csv(src)
    vecs = np.vstack(
        [np.array(ast.literal_eval(s), dtype=np.float32) for s in df["embedding_text"]]
    )
    assert vecs.shape == (len(df), DIM), vecs.shape

    meta = df.drop(columns=["embedding_text"]).reset_index(drop=True)
    np.save(OUT / "facebook_embeddings.npy", vecs)
    meta.to_parquet(OUT / "facebook_meta.parquet")
    norms = np.linalg.norm(vecs, axis=1)
    print(
        f"facebook: {vecs.shape} float32, "
        f"normes min/med/max = {norms.min():.4f}/{np.median(norms):.4f}/{norms.max():.4f}",
        flush=True,
    )
    print(f"{meta['post_owner.name'].nunique()} organisations, {len(meta)} posts", flush=True)


if __name__ == "__main__":
    main()
