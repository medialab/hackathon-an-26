#!/usr/bin/env python3
"""Carte globale enrichie : maximum d'activités HATVP + optimisations gold.

- Réutilise la UMAP conjointe globale (umap_coords.npz)
- Liens gold avec contrainte d'antériorité d'exercice + snippet justificatif
- Amendements gold marqués distinctement (flag a_gold)
- Pour chaque paire gold, top-K vrais voisins en 1536-d :
    nn_a = amendements non gold les plus proches de l'amendement attesté
  (candidats à de nouveaux liens — la proximité 2D UMAP peut mentir, pas celle-ci)
- Amendements "égaux non déclarés" (data/equalitiesNonDeclare.json) : même texte
  qu'un amendement gold mais sans citation du lobby → flag a_gold=2 + mapping a_eq
- a_full : exposé sommaire intégral (tous en version complète, sinon
  gold ∪ égaux ∪ voisins) pour vérifier les prédictions au clic
- Posts Facebook (subset_facebook_posts_matched.csv, via convert_facebook.py) :
  organisme résolu au répertoire HATVP (normalisation + alias manuels pour les
  quelques dénominations qui ne matchent pas telles quelles), lien vers
  l'activité de cet organisme au cosinus max sous contrainte d'antériorité
  d'exercice (repli sans contrainte si aucune activité antérieure). Pour
  chaque post, top-K voisins 1536-d toutes orgs confondues (fb_nn) : candidats
  exploratoires même quand l'organisme n'est pas recensé à la HATVP.

Usage : build_full_map.py [n_sample_hatvp]   (défaut : toutes les activités)
Sortie : data/embeddings/full_map_data.json
"""
import base64
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EMB = ROOT / "data" / "embeddings"

N_DOMAINS = 8
K_NN = 5

rng = np.random.default_rng(42)

# Dénominations Facebook (post_owner.name) qui ne matchent pas la HATVP telles
# quelles une fois normalisées (accents/casse/ponctuation) — alias manuels vers
# la dénomination HATVP réelle. Repérés par recherche exacte + fuzzy (difflib)
# une fois toutes les autres orgs résolues automatiquement.
FB_ORG_ALIASES = {
    "Association pour le Droit de Mourir dans la Dignité - ADMD": "ASS POUR DROIT MOURIR DIGNITE",
    "Chambres d'agriculture": "CHAMBRES D'AGRICULTURE FRANCE",
    "Jeunes Agriculteurs Syndicat": "JEUNES AGRICULTEURS",
    "La Ligue contre le cancer": "LIGUE NATIONALE CONTRE LE CANCER",
    "Les bouchers, bouchers-charcutiers de France":
        "CONFEDERATION FRANCAISE DE LA BOUCHERIE - BOUCHERIE CHARCUTERIE - TRAITEURS",
    "Réseau Action Climat": "RESEAU ACTION CLIMAT FRANCE",
    "SDI Syndicat des Indépendants et des TPE": "SYNDICAT DES INDEPENDANTS",
    "Secours Catholique - Caritas France": "SECOURS CATHOLIQUE",
    "UNICEF France": "COMITE FRANCAIS POUR L'UNICEF",
    # Non recensées au répertoire HATVP (vérifié) : Adie, Face à l'inceste.
}


def b64_f32(a): return base64.b64encode(np.ascontiguousarray(a, np.float32).tobytes()).decode()
def b64_u8(a): return base64.b64encode(np.ascontiguousarray(a, np.uint8).tobytes()).decode()


def trunc(s, n):
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def norm_org(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9 ]", " ", s.upper())
    return re.sub(r"\s+", " ", s).strip()


def fmt_int(n) -> str:
    try:
        return f"{int(n):,}".replace(",", " ")
    except (TypeError, ValueError):
        return "0"


def main() -> None:
    n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else None

    coords = np.load(EMB / "umap_coords.npz")
    ch_all, ca = coords["hatvp"], coords["amendments"]
    hm = pd.read_parquet(EMB / "hatvp_meta.parquet")
    am = pd.read_parquet(EMB / "amendments_meta.parquet")
    ex = pd.read_parquet(EMB / "hatvp_exercices.parquet")
    he = np.load(EMB / "hatvp_embeddings.npy")
    ae = np.load(EMB / "amendments_embeddings.npy")
    he /= np.linalg.norm(he, axis=1, keepdims=True)
    ae /= np.linalg.norm(ae, axis=1, keepdims=True)

    gold_csv = pd.read_csv(ROOT / "data" / "gold" / "gold_hatvp.csv")
    snip = gold_csv.set_index("row_id")[["snippet", "evidence"]]

    denoms = hm["denomination"].values
    ex_debut = ex["ex_debut"].values
    amd_gold = np.where((am["org_denomination"].str.strip() != "").values)[0]
    gold_set = set(int(i) for i in amd_gold)

    # --- liens gold : max cos parmi les activités d'exercice antérieur ---
    links = []
    for i in amd_gold:
        org = am["org_denomination"].iat[i].strip()
        rows = np.where(denoms == org)[0]
        if len(rows) == 0:
            continue
        ok = rows[(ex_debut[rows] != "") & (ex_debut[rows] <= am["date"].iat[i])]
        if len(ok) == 0:
            continue
        sims = he[ok] @ ae[i]
        links.append((int(i), int(ok[np.argmax(sims)]), float(sims.max())))
    print(f"{len(links)} liens gold (antériorité OK)", flush=True)

    # --- vrais voisins 1536-d de chaque paire gold ---
    la = np.array([l[0] for l in links])
    lh = np.array([l[1] for l in links])
    sims_a = ae @ ae[la].T                          # (13405, n_links)
    sims_a[la, :] = -1                              # exclut les amendements gold
    for k in gold_set:
        sims_a[k, :] = -1
    nn_a = []
    for c in range(sims_a.shape[1]):
        top = np.argpartition(-sims_a[:, c], K_NN)[:K_NN]
        top = top[np.argsort(-sims_a[top, c])]
        nn_a.append([[int(t), round(float(sims_a[t, c]), 3)] for t in top])

    # --- amendements égaux non déclarés (même texte, pas de citation) ---
    import json as _json
    eq_path = ROOT / "data" / "equalitiesNonDeclare.json"
    if eq_path.exists():
        eqs = _json.loads(eq_path.read_text())
    else:
        print(f"⚠ {eq_path} introuvable — jumeaux non déclarés désactivés", flush=True)
        eqs = {}
    rowid_pos = {str(r): k for k, r in enumerate(am["row_id"])}
    index_pos = {str(r): k for k, r in enumerate(am["index"])}
    a_eq: dict[int, list[int]] = {}
    for rid, twins in eqs.items():
        gp = rowid_pos.get(str(rid))
        if gp is None:
            continue
        for t in twins:
            tp = index_pos.get(str(t))
            if tp is not None and tp != gp:
                a_eq.setdefault(tp, []).append(gp)
    print(f"{len(a_eq)} amendements égaux non déclarés (jumeaux de gold)", flush=True)

    # --- posts Facebook : résolution de l'organisme + lien vers activité ---
    cf = coords["facebook"]
    fm = pd.read_parquet(EMB / "facebook_meta.parquet")
    fm["text"] = fm["text"].fillna("")
    fm["text_short"] = fm["text_short"].fillna(fm["text"])
    fe = np.load(EMB / "facebook_embeddings.npy")
    fe /= np.linalg.norm(fe, axis=1, keepdims=True)

    denom_by_norm: dict[str, str] = {}
    for d in denoms:
        denom_by_norm.setdefault(norm_org(d), d)
    fb_org = [denom_by_norm.get(norm_org(FB_ORG_ALIASES.get(o, o)))
              for o in fm["post_owner.name"]]
    n_matched = sum(o is not None for o in fb_org)
    print(f"Facebook : {n_matched}/{len(fm)} posts avec organisme résolu à la HATVP "
          f"({fm['post_owner.name'].nunique()} orgs)", flush=True)

    fb_links, fb_temporal = [], []
    for i, org in enumerate(fb_org):
        if org is None:
            continue
        rows = np.where(denoms == org)[0]
        if len(rows) == 0:
            continue
        post_date = str(fm["creation_time"].iat[i])[:10]
        ok = rows[(ex_debut[rows] != "") & (ex_debut[rows] <= post_date)]
        temporal_ok = 1
        if len(ok) == 0:  # repli : pas d'activité antérieure, on ignore la contrainte
            ok, temporal_ok = rows, 0
        sims = he[ok] @ fe[i]
        fb_links.append((i, int(ok[np.argmax(sims)]), float(sims.max())))
        fb_temporal.append(temporal_ok)
    print(f"{len(fb_links)} liens post→activité (organisme identifié)", flush=True)

    # --- voisins 1536-d de chaque post, toutes orgs confondues (exploratoire) ---
    sims_fb = he @ fe.T  # (n_hatvp, n_fb)
    nn_fb = []
    for c in range(sims_fb.shape[1]):
        top = np.argpartition(-sims_fb[:, c], K_NN)[:K_NN]
        top = top[np.argsort(-sims_fb[top, c])]
        nn_fb.append([[int(t), round(float(sims_fb[t, c]), 3)] for t in top])

    # --- échantillon : tout par défaut, sinon uniforme + points forcés ---
    forced = sorted({j for _, j, _ in links} | {j for _, j, _ in fb_links})
    if n_sample is None or n_sample >= len(hm):
        hidx = np.arange(len(hm))
    else:
        pool = np.setdiff1d(np.arange(len(hm)), np.array(forced))
        picked = rng.choice(pool, size=n_sample, replace=False)
        hidx = np.sort(np.concatenate([np.array(forced), picked]))
    hpos = {int(j): k for k, j in enumerate(hidx)}

    # --- catégories domaines (tous les domaines de la ligne) ---
    dom_lists = hm["domaines"].fillna("").str.split("§")
    top = (dom_lists.str[0].replace("", "Non renseigné").iloc[hidx]
           .value_counts().head(N_DOMAINS).index.tolist())
    rank = {d: k for k, d in enumerate(top)}
    cat = np.full(len(hidx), N_DOMAINS, dtype=np.uint8)
    for pos, i in enumerate(hidx):
        cat[pos] = min((rank[d] for d in dom_lists.iat[i] if d in rank), default=N_DOMAINS)

    # --- textes ---
    h_texts = []
    for j in hidx:
        r = hm.iloc[j]
        per = f"exercice {ex_debut[j][:4]} · décl. {r['date'][:7]}" if ex_debut[j] else r["date"][:7]
        h_texts.append([trunc(r["denomination"], 45), per,
                        trunc((r["domaines"] or "").split("§")[0], 32), r["objet"].strip()])
    a_texts = [[trunc(r.dossier, 65), f"{r.author_name} ({r.author_group})", r.date[:10],
                trunc(r.amendment_summary, 250), trunc(r.org_denomination.strip(), 55)]
               for r in am.itertuples()]
    a_gold = np.zeros(len(am), dtype=np.uint8)
    a_gold[list(a_eq.keys())] = 2
    a_gold[amd_gold] = 1

    # exposés intégraux : tous si carte complète, sinon gold ∪ égaux ∪ voisins
    if n_sample is None or n_sample >= len(hm):
        full_set = range(len(am))
    else:  # version artifact allégée : intégral pour gold + jumeaux seulement
        full_set = set(map(int, amd_gold)) | set(a_eq)
    a_full = {str(i): am["amendment_summary"].iat[i].strip() for i in full_set}

    g_amd, g_act, out_links, out_nn_a = [], [], [], []
    for k, (a, j, c) in enumerate(links):
        ra = am.iloc[a]
        s = snip.loc[int(ra["row_id"])]
        g_amd.append([f"n° {ra['number']}", ra["amendment_content"].strip(),
                      ra["amendment_summary"].strip(), ra["match_type"], str(ra["score"]),
                      str(s["snippet"]), str(s["evidence"])])
        rh = hm.iloc[j]
        g_act.append([f"exercice {ex_debut[j]} → {ex['ex_fin'].iat[j]} · déclaré le {rh['date'][:10]}",
                      rh["domaines"], rh["actions"], rh["decisions"],
                      rh["responsables"], rh["tiers"]])
        out_links.append([int(a), hpos[j], round(c, 3)])
        out_nn_a.append(nn_a[k])

    # --- textes des posts Facebook (accès par colonne : noms avec points,
    # incompatibles avec itertuples) ---
    fb_texts = []
    for i in range(len(fm)):
        date = str(fm["creation_time"].iat[i])[:10]
        stats = (f"{fmt_int(fm['statistics.reaction_count'].iat[i])} réactions · "
                 f"{fmt_int(fm['statistics.comment_count'].iat[i])} commentaires · "
                 f"{fmt_int(fm['statistics.share_count'].iat[i])} partages")
        org_disp = fb_org[i] or f"{fm['post_owner.name'].iat[i]} (org. non recensée HATVP)"
        fb_texts.append([org_disp, date, trunc(fm["text_short"].iat[i], 260), stats,
                          str(fm["mcl_url"].iat[i] or "")])
    fb_full = {str(i): (fm["text"].iat[i] or "").strip() for i in range(len(fm))}
    out_fb_links, out_fb_act = [], []
    for (i, j, c), t in zip(fb_links, fb_temporal):
        if j not in hpos:
            continue
        out_fb_links.append([i, hpos[j], round(c, 3), t])
        rh = hm.iloc[j]
        out_fb_act.append([f"exercice {ex_debut[j]} → {ex['ex_fin'].iat[j]} · déclaré le {rh['date'][:10]}",
                            rh["domaines"], rh["actions"], rh["decisions"],
                            rh["responsables"], rh["tiers"]])
    out_nn_fb = [[[hpos[t], c] for t, c in row if t in hpos] for row in nn_fb]

    out = {
        "stats": (f"{len(hidx):,} activités HATVP ({'toutes' if len(hidx)==len(hm) else f'sur {len(hm):,}'}) · "
                  f"{len(am):,} amendements dont {len(amd_gold)} gold et {len(a_eq)} jumeaux non déclarés · "
                  f"{len(out_links)} liens gold (exercice antérieur) · "
                  f"{len(fm):,} posts Facebook ({n_matched} avec organisme identifié · "
                  f"{len(out_fb_links)} liens vers une activité)").replace(",", " "),
        "domains": top + ["Autres"],
        "h_xy": b64_f32(ch_all[hidx]), "h_cat": b64_u8(cat), "h_txt": h_texts,
        "a_xy": b64_f32(ca), "a_txt": a_texts, "a_gold": b64_u8(a_gold),
        "links": out_links, "g_amd": g_amd, "g_act": g_act,
        "nn_a": out_nn_a,
        "a_eq": {str(k): v for k, v in a_eq.items()},
        "a_full": a_full,
        "fb_xy": b64_f32(cf), "fb_txt": fb_texts, "fb_full": fb_full,
        "fb_links": out_fb_links, "fb_act": out_fb_act, "fb_nn": out_nn_fb,
        "n_hatvp_total": int(len(hm)), "n_amd": int(len(am)), "n_fb": int(len(fm)),
    }
    p = EMB / (f"full_map_data_{n_sample}.json" if n_sample and n_sample < len(hm)
               else "full_map_data.json")
    p.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"{p} : {p.stat().st_size/1e6:.1f} Mo, {len(hidx)} activités", flush=True)


if __name__ == "__main__":
    main()
