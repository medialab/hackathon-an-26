#!/usr/bin/env python3
"""Récupère la période d'exercice (dateDebut/dateFin) de chaque activité HATVP.

La colonne `date` d'objets_embed.csv est la date de PUBLICATION de la fiche
(pic en mars = échéance déclarative annuelle), pas la date de l'action.
La vraie fenêtre temporelle est l'exercice parent dans le JSON source.

Jointure sur (denomination, date de publication, début d'objet) puis sortie
alignée sur les lignes de hatvp_meta.parquet :
  data/embeddings/hatvp_exercices.parquet  (ex_debut, ex_fin, ISO ou "")
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EMB = ROOT / "data" / "embeddings"

WS = re.compile(r"\s+")


def norm(s: str) -> str:
    return WS.sub(" ", (s or "").strip())[:80]


def iso_dmy(s: str) -> str:  # "01-01-2023" -> "2023-01-01"
    d, m, y = s.split("-")
    return f"{y}-{m}-{d}"


def iso_pub(s: str) -> str:  # "29/04/2024 à 16:46:10" -> "2024-04-29T16:46:10"
    date, time = s.split(" à ")
    d, m, y = date.split("/")
    return f"{y}-{m}-{d}T{time}"


def main() -> None:
    with open(ROOT / "data" / "agora_repertoire_opendata.json") as f:
        data = json.load(f)

    index: dict[tuple, tuple] = {}
    for pub in data["publications"]:
        denom = norm(pub.get("denomination", ""))
        for ex in pub.get("exercices", []):
            pc = ex.get("publicationCourante") or {}
            debut, fin = pc.get("dateDebut"), pc.get("dateFin")
            if not debut:
                continue
            for act in pc.get("activites") or []:
                a = act.get("publicationCourante") or {}
                key = (denom, iso_pub(a.get("publicationDate", "01/01/1970 à 00:00:00")),
                       norm(a.get("objet", "")))
                index[key] = (iso_dmy(debut), iso_dmy(fin) if fin else "")

    hm = pd.read_parquet(EMB / "hatvp_meta.parquet")
    debuts, fins, miss = [], [], 0
    for r in hm.itertuples():
        key = (norm(r.denomination), r.date, norm(r.objet))
        d, f_ = index.get(key, ("", ""))
        if not d:
            miss += 1
        debuts.append(d)
        fins.append(f_)
    out = pd.DataFrame({"ex_debut": debuts, "ex_fin": fins})
    out.to_parquet(EMB / "hatvp_exercices.parquet")
    print(f"{len(hm)} activités, exercice retrouvé pour {len(hm)-miss} ({miss} manquants)")


if __name__ == "__main__":
    main()
