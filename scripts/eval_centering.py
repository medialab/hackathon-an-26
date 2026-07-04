#!/usr/bin/env python3
"""Évalue le recentrage par corpus (retrait du vecteur moyen de chaque corpus).

Métrique qui compte : pour chaque amendement gold, rang de l'org gold parmi
toutes les orgs du répertoire (score org = max cosinus sur ses activités).
Le biais de registre gonfle une composante commune aux deux corpus ; la retirer
doit resserrer le classement. On compare brut vs centré.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EMB = ROOT / "data" / "embeddings"


def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def eval_ranks(he, ae, gold_idx, gold_orgs, org_codes, n_orgs):
    sims = he @ ae[gold_idx].T                      # (n_act, n_gold)
    df = pd.DataFrame(sims)
    df["org"] = org_codes
    per_org = df.groupby("org").max().values        # (n_orgs, n_gold)
    ranks, cos_gold = [], []
    for k, oc in enumerate(gold_orgs):
        col = per_org[:, k]
        s = col[oc]
        ranks.append(int((col > s).sum()) + 1)
        cos_gold.append(float(s))
    r = np.array(ranks)
    return r, np.array(cos_gold)


def main():
    he = l2(np.load(EMB / "hatvp_embeddings.npy"))
    ae = l2(np.load(EMB / "amendments_embeddings.npy"))
    hm = pd.read_parquet(EMB / "hatvp_meta.parquet")
    am = pd.read_parquet(EMB / "amendments_meta.parquet")

    denoms = pd.Series(hm["denomination"].values)
    org_codes, org_names = pd.factorize(denoms)
    name_pos = {n: i for i, n in enumerate(org_names)}

    gold_idx, gold_orgs = [], []
    for i in np.where((am["org_denomination"].str.strip() != "").values)[0]:
        org = am["org_denomination"].iat[i].strip()
        if org in name_pos:
            gold_idx.append(i)
            gold_orgs.append(name_pos[org])
    print(f"{len(gold_idx)} amendements gold évaluables, {len(org_names)} orgs au total\n")

    for label, hx, ax in [
        ("brut", he, ae),
        ("centré par corpus", l2(he - he.mean(0)), l2(ae - ae.mean(0))),
    ]:
        r, cg = eval_ranks(hx, ax, gold_idx, gold_orgs, org_codes, len(org_names))
        print(f"[{label}]")
        print(f"  rang org gold : med={int(np.median(r))} p25={int(np.percentile(r,25))} "
              f"p75={int(np.percentile(r,75))}")
        print(f"  recall@1={np.mean(r<=1):.1%}  @10={np.mean(r<=10):.1%}  "
              f"@50={np.mean(r<=50):.1%}  @100={np.mean(r<=100):.1%}")
        print(f"  cos gold : med={np.median(cg):.3f}\n")


if __name__ == "__main__":
    main()
