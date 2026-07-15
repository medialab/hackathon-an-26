#!/usr/bin/env python3
"""Projection UMAP conjointe HATVP + amendements + posts Facebook, et
vérification du même espace.

Entrées : data/embeddings/*.npy + *.parquet
          (cf. convert_embeddings.py, convert_facebook.py)
Sorties : data/embeddings/umap_coords.npz  (coords 2D des trois jeux)
          stdout : sanity check paires gold vs aléatoire
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EMB = ROOT / "data" / "embeddings"

rng = np.random.default_rng(42)


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main() -> None:
    hatvp = l2norm(np.load(EMB / "hatvp_embeddings.npy"))
    amd = l2norm(np.load(EMB / "amendments_embeddings.npy"))
    fb = l2norm(np.load(EMB / "facebook_embeddings.npy"))
    hatvp_meta = pd.read_parquet(EMB / "hatvp_meta.parquet")
    amd_meta = pd.read_parquet(EMB / "amendments_meta.parquet")
    print(f"HATVP {hatvp.shape}, amendements {amd.shape}, Facebook {fb.shape}", flush=True)

    # --- Sanity check : les paires gold doivent être plus proches que le hasard ---
    gold = amd_meta[amd_meta["org_denomination"].str.strip() != ""]
    denoms = hatvp_meta["denomination"].values
    sims_gold, sims_rand = [], []
    all_orgs = pd.unique(denoms)
    for idx, row in gold.head(150).iterrows():
        org = row["org_denomination"].strip()
        mask = denoms == org
        if not mask.any():
            continue
        a = amd[amd_meta.index.get_loc(idx)]
        sims_gold.append(float((hatvp[mask] @ a).max()))
        rand_org = rng.choice(all_orgs)
        rmask = denoms == rand_org
        sims_rand.append(float((hatvp[rmask] @ a).max()))
    print(
        f"cos max amendement→org gold   : med={np.median(sims_gold):.3f} (n={len(sims_gold)})\n"
        f"cos max amendement→org random : med={np.median(sims_rand):.3f}",
        flush=True,
    )

    # --- PCA 50 sur l'ensemble, puis UMAP 2D ---
    from sklearn.decomposition import PCA

    X = np.vstack([hatvp, amd, fb])
    pca = PCA(n_components=50, random_state=42)
    Xp = pca.fit_transform(X)
    print(f"PCA 50 : variance expliquée {pca.explained_variance_ratio_.sum():.2%}", flush=True)

    import umap

    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42, verbose=True
    )
    coords = reducer.fit_transform(Xp).astype(np.float32)
    n_h, n_a = hatvp.shape[0], amd.shape[0]
    np.savez_compressed(
        EMB / "umap_coords.npz",
        hatvp=coords[:n_h],
        amendments=coords[n_h:n_h + n_a],
        facebook=coords[n_h + n_a:],
    )
    print("UMAP OK →", EMB / "umap_coords.npz", flush=True)


if __name__ == "__main__":
    main()
