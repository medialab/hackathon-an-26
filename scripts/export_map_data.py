#!/usr/bin/env python3
"""Exporte les données de la carte UMAP vers un JSON compact pour l'artifact.

- Échantillonne les activités HATVP (25 000 + toutes celles liées aux paires gold)
- Garde tous les amendements
- Pour chaque paire gold amendement→org, retrouve l'activité de l'org la plus
  proche en cosinus (le lien tracé sur la carte)

Sortie : data/embeddings/map_data.json
"""
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EMB = ROOT / "data" / "embeddings"

SAMPLE_HATVP = 25_000
N_DOMAINS = 8  # top-N domaines colorés, le reste en "Autres"

rng = np.random.default_rng(42)


def b64_f32(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()


def b64_u8(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode()


def trunc(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def main() -> None:
    coords = np.load(EMB / "umap_coords.npz")
    ch, ca = coords["hatvp"], coords["amendments"]
    hm = pd.read_parquet(EMB / "hatvp_meta.parquet")
    am = pd.read_parquet(EMB / "amendments_meta.parquet")
    he = np.load(EMB / "hatvp_embeddings.npy")
    ae = np.load(EMB / "amendments_embeddings.npy")
    he /= np.linalg.norm(he, axis=1, keepdims=True)
    ae /= np.linalg.norm(ae, axis=1, keepdims=True)

    # --- paires gold : meilleure activité de l'org appariée ---
    denoms = hm["denomination"].values
    gold_links = []  # (amd_row, hatvp_row, cos)
    gold_mask = am["org_denomination"].str.strip() != ""
    for i in np.where(gold_mask)[0]:
        org = am["org_denomination"].iat[i].strip()
        rows = np.where(denoms == org)[0]
        if len(rows) == 0:
            continue
        sims = he[rows] @ ae[i]
        j = rows[int(np.argmax(sims))]
        gold_links.append((int(i), int(j), float(sims.max())))

    # --- échantillon HATVP : uniforme + activités gold forcées ---
    forced = np.array(sorted({j for _, j, _ in gold_links}), dtype=int)
    pool = np.setdiff1d(np.arange(len(hm)), forced)
    picked = rng.choice(pool, size=min(SAMPLE_HATVP, len(pool)), replace=False)
    hidx = np.sort(np.concatenate([forced, picked]))
    hpos = {int(j): k for k, j in enumerate(hidx)}  # index origine -> index viz

    # --- catégories : top-N domaines + Autres ---
    # top-N sur le domaine principal, mais l'affectation regarde tous les
    # domaines de la ligne (sinon plus de la moitié tombe dans "Autres")
    dom_lists = hm["domaines"].fillna("").str.split("§")
    top = (
        dom_lists.str[0].replace("", "Non renseigné").iloc[hidx].value_counts()
        .head(N_DOMAINS).index.tolist()
    )
    rank = {d: k for k, d in enumerate(top)}
    cat = np.full(len(hidx), N_DOMAINS, dtype=np.uint8)  # N_DOMAINS = Autres
    for pos, i in enumerate(hidx):
        best = min((rank[d] for d in dom_lists.iat[i] if d in rank), default=N_DOMAINS)
        cat[pos] = best

    # --- textes de survol (objet HATVP complet : quasi gratuit, médiane 123 car.) ---
    hs = hm.iloc[hidx]
    h_texts = [
        [trunc(r.denomination, 55), r.date[:4], trunc(r.domaines.split("§")[0] if r.domaines else "", 40), r.objet.strip()]
        for r in hs.itertuples()
    ]
    a_texts = [
        [trunc(r.dossier, 65), f"{r.author_name} ({r.author_group})", r.date[:10],
         trunc(r.amendment_summary, 400), trunc(r.org_denomination.strip(), 55)]
        for r in am.itertuples()
    ]

    links = [[a, hpos[j], round(c, 3)] for a, j, c in gold_links]

    # --- détail intégral des paires gold (panneau latéral) ---
    g_amd, g_act = [], []
    for a, j, _ in gold_links:
        ra, rh = am.iloc[a], hm.iloc[j]
        g_amd.append([
            f"n° {ra['number']}", ra["amendment_content"].strip(), ra["amendment_summary"].strip(),
            ra["match_type"], str(ra["score"]),
        ])
        g_act.append([
            rh["date"][:10], rh["domaines"], rh["actions"], rh["decisions"],
            rh["responsables"], rh["tiers"],
        ])

    out = {
        "domains": top + ["Autres"],
        "h_xy": b64_f32(ch[hidx]),
        "h_cat": b64_u8(cat),
        "h_txt": h_texts,
        "a_xy": b64_f32(ca),
        "a_txt": a_texts,
        "links": links,
        "g_amd": g_amd,
        "g_act": g_act,
        "n_hatvp_total": int(len(hm)),
        "n_amd": int(len(am)),
    }
    p = EMB / "map_data.json"
    p.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"{p} : {p.stat().st_size/1e6:.1f} Mo, {len(hidx)} pts HATVP, "
          f"{len(am)} amendements, {len(links)} liens gold", flush=True)


if __name__ == "__main__":
    main()
