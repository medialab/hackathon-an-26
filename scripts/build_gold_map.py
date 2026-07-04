#!/usr/bin/env python3
"""Carte focalisée sur les acteurs gold : activités des orgs citées dans les
exposés d'amendements + les 489 amendements gold, UMAP dédiée.

Liens : pour chaque amendement gold, meilleure activité en cosinus de l'org
appariée PARMI les activités temporellement possibles (exercice commencé avant
la date de l'amendement — la date de publication de la fiche est une date de
déclaration, souvent postérieure à l'action).

Sortie : data/embeddings/gold_map_data.json
"""
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EMB = ROOT / "data" / "embeddings"

N_ORGS_COLORED = 8


def b64_f32(a): return base64.b64encode(np.ascontiguousarray(a, np.float32).tobytes()).decode()
def b64_u8(a): return base64.b64encode(np.ascontiguousarray(a, np.uint8).tobytes()).decode()


def trunc(s, n):
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def main() -> None:
    hm = pd.read_parquet(EMB / "hatvp_meta.parquet")
    am = pd.read_parquet(EMB / "amendments_meta.parquet")
    ex = pd.read_parquet(EMB / "hatvp_exercices.parquet")
    he = np.load(EMB / "hatvp_embeddings.npy")
    ae = np.load(EMB / "amendments_embeddings.npy")
    he /= np.linalg.norm(he, axis=1, keepdims=True)
    ae /= np.linalg.norm(ae, axis=1, keepdims=True)

    gold_csv = pd.read_csv(ROOT / "data" / "gold" / "gold_hatvp.csv")
    snip = gold_csv.set_index("row_id")[["snippet", "evidence", "source_field"]]

    amd_idx = np.where((am["org_denomination"].str.strip() != "").values)[0]
    orgs = sorted({am["org_denomination"].iat[i].strip() for i in amd_idx})
    denoms = hm["denomination"].values
    act_mask = np.isin(denoms, orgs)
    act_idx = np.where(act_mask)[0]
    print(f"{len(amd_idx)} amendements gold, {len(orgs)} orgs, {len(act_idx)} activités", flush=True)

    # --- liens avec contrainte d'antériorité (exercice commencé avant l'amendement) ---
    ex_debut = ex["ex_debut"].values
    links, no_temporal = [], 0
    act_pos = {int(j): k for k, j in enumerate(act_idx)}
    for i in amd_idx:
        org = am["org_denomination"].iat[i].strip()
        amd_date = am["date"].iat[i]
        rows = act_idx[denoms[act_idx] == org]
        if len(rows) == 0:
            continue
        ok = rows[(ex_debut[rows] != "") & (ex_debut[rows] <= amd_date)]
        if len(ok) == 0:
            no_temporal += 1
            continue
        sims = he[ok] @ ae[i]
        j = int(ok[np.argmax(sims)])
        links.append((int(i), j, float(sims.max())))
    print(f"{len(links)} liens (antériorité OK), {no_temporal} sans activité antérieure", flush=True)

    # --- UMAP dédiée au sous-ensemble ---
    from sklearn.decomposition import PCA
    import umap

    X = np.vstack([he[act_idx], ae[amd_idx]])
    Xp = PCA(n_components=50, random_state=42).fit_transform(X)
    coords = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                       random_state=42, verbose=True).fit_transform(Xp).astype(np.float32)
    ch, ca = coords[: len(act_idx)], coords[len(act_idx):]
    amd_pos = {int(i): k for k, i in enumerate(amd_idx)}

    # --- catégories : top orgs par nb de liens gold ---
    per_org = pd.Series([am["org_denomination"].iat[a].strip() for a, _, _ in links]).value_counts()
    top = per_org.head(N_ORGS_COLORED).index.tolist()
    rank = {o: k for k, o in enumerate(top)}
    cat = np.array([rank.get(denoms[j], N_ORGS_COLORED) for j in act_idx], dtype=np.uint8)

    # --- textes ---
    h_texts = []
    for j in act_idx:
        r = hm.iloc[j]
        per = f"exercice {ex_debut[j][:4]}" if ex_debut[j] else "exercice inconnu"
        h_texts.append([trunc(r["denomination"], 55), f"{per} · déclaré {r['date'][:10]}",
                        trunc((r["domaines"] or "").split("§")[0], 40), r["objet"].strip()])
    a_texts = []
    for i in amd_idx:
        r = am.iloc[i]
        a_texts.append([trunc(r["dossier"], 65), f"{r['author_name']} ({r['author_group']})",
                        r["date"][:10], trunc(r["amendment_summary"], 400),
                        trunc(r["org_denomination"].strip(), 55)])

    g_amd, g_act, out_links = [], [], []
    for a, j, c in links:
        ra = am.iloc[a]
        s = snip.loc[int(ra["row_id"])]
        g_amd.append([f"n° {ra['number']}", ra["amendment_content"].strip(),
                      ra["amendment_summary"].strip(), ra["match_type"], str(ra["score"]),
                      str(s["snippet"]), str(s["evidence"])])
        rh = hm.iloc[j]
        g_act.append([f"exercice {ex_debut[j]} → {ex['ex_fin'].iat[j]} · déclaré le {rh['date'][:10]}",
                      rh["domaines"], rh["actions"], rh["decisions"],
                      rh["responsables"], rh["tiers"]])
        out_links.append([amd_pos[a], act_pos[j], round(c, 3)])

    out = {
        "gold": True,
        "cat_label": "Orgs les plus citées (nb d'activités)",
        "stats": (f"{len(act_idx):,} activités déclarées par les {len(orgs)} organisations citées · "
                  f"{len(amd_idx)} amendements gold · {len(out_links)} liens (exercice antérieur à l’amendement)"
                  ).replace(",", " "),
        "domains": [trunc(o.title() if o.isupper() else o, 34) for o in top] + ["Autres orgs"],
        "h_xy": b64_f32(ch), "h_cat": b64_u8(cat), "h_txt": h_texts,
        "a_xy": b64_f32(ca), "a_txt": a_texts,
        "links": out_links, "g_amd": g_amd, "g_act": g_act,
        "n_hatvp_total": int(len(hm)), "n_amd": int(len(amd_idx)),
    }
    p = EMB / "gold_map_data.json"
    p.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"{p} : {p.stat().st_size/1e6:.1f} Mo", flush=True)


if __name__ == "__main__":
    main()
