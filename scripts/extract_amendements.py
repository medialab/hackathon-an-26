#!/usr/bin/env python3
"""Extraction de TOUS les amendements de l'API CLAIR.

L'API n'expose pas d'endpoint global "tous les amendements" : ils sont
accessibles par dossier législatif et par parlementaire. Le script combine
les deux routes puis déduplique par uid :

  1. /dossiers/{uid}/amendements       -> couvre les amendements rattachés à un
                                          dossier, y compris ceux du Gouvernement
  2. /parlementaires/{slug}/amendements -> couvre les amendements hors dossier et
                                          fournit le mapping amendement -> signataires

Chaque amendement est enrichi avec le groupe politique (sigle, couleur,
position) de son auteur et de chacun de ses signataires.

Sorties (dans --out, data/ par défaut) :
  - amendements.jsonl      un amendement par ligne, dédupliqué, avec `signataires`
                           (slug + groupe + couleur) et `auteur` (nom + groupe)
  - amendements.csv        version aplatie pour tableur / pandas
  - dossiers.jsonl         référentiel des dossiers législatifs
  - parlementaires.jsonl   référentiel des parlementaires (fiches complètes)
  - groupes.jsonl          référentiel des groupes politiques (couleur, position)
  - extraction_meta.json   date, volumétrie, paramètres de l'extraction

Reprise sur interruption : l'état est sauvegardé dans <out>/_state/ ; relancer
la même commande reprend là où ça s'était arrêté. --fresh repart de zéro.

Lancement (depuis la racine du repo) :

  # 1. renseigner la clé API (une seule fois) — sinon rate-limit sévère
  cp .env.example .env       # puis remplir CLAIR_API_KEY=... (demander à Axel)

  # 2. extraction complète (~187 000 amendements, ~15-25 min, ~1 Go dans data/)
  python3 scripts/extract_amendements.py

  # variante : test rapide sur un échantillon (3 dossiers + 3 parlementaires)
  python3 scripts/extract_amendements.py --out data/sample --max-entities 3

Aucune dépendance hors bibliothèque standard (Python >= 3.9).
"""

import argparse
import csv
import json
import os
import re
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "https://clair-production.up.railway.app/api/v1"
PAGE_LIMIT = 100          # maximum accepté par l'API (validation zod)
PAGE_LIMIT_PARL = 50      # maximum sur /parlementaires/{slug}/amendements
PAGE_WORKERS = 5          # pages téléchargées en parallèle pour une même entité
MAX_RETRIES = 8

CSV_COLUMNS = [
    "uid", "numero", "legislature", "chambre", "texteRef", "articleVise",
    "sort", "dateDepot", "dateSort", "auteurLibelle", "auteur_nom",
    "auteur_groupe", "auteur_couleur", "signataires", "signataires_groupes",
    "dossier_uid", "dossier_titre", "nb_scrutins", "exposeSommaire", "dispositif",
]

CIVILITE_RE = re.compile(r"^(m\.|mme|mm\.|mmes|m)\s+", re.IGNORECASE)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def normalize(name: str) -> str:
    """minuscules + sans accents, pour comparer des noms propres."""
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def split_names(auteur_libelle: str) -> list:
    """Découpe 'Mme A, M. B et M. C' en noms individuels.

    À l'Assemblée les listes sont complètes (vérifié : jamais de « et
    plusieurs de ses collègues ») ; au Sénat le libellé ne contient souvent
    que le premier auteur. Les mentions « rapporteur(e) » cassent parfois la
    ponctuation ('rapporteur Mme X'), on les remplace par un séparateur."""
    cleaned = re.sub(r",?\s*rapporteure?s?(\s+génér\w+)?\s*", ", ",
                     auteur_libelle or "", flags=re.IGNORECASE)
    return [n for n in (s.strip() for s in re.split(r",| et ", cleaned)) if n]


class ApiClient:
    """GET JSON avec retry/backoff (429, 5xx, erreurs réseau)."""

    def __init__(self, api_key):
        self.api_key = api_key
        self.requests_done = 0
        self._lock = threading.Lock()

    def get(self, path: str, **params) -> dict:
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{BASE_URL}{path}" + (f"?{query}" if query else "")
        headers = {"Accept": "application/json", "User-Agent": "hackathon-an-26-extractor"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.load(resp)
                with self._lock:
                    self.requests_done += 1
                return data
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code == 429 or e.code >= 500:
                    retry_after = e.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after else min(2 ** attempt, 60)
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"HTTP {e.code} sur {url}: {e.read()[:300]!r}") from e
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last_err = e
                time.sleep(min(2 ** attempt, 60))
        raise RuntimeError(f"Échec après {MAX_RETRIES} tentatives sur {url}: {last_err}")

    def get_all_pages(self, path: str, limit: int, first: dict = None, **params) -> list:
        """Toutes les pages d'un endpoint paginé {data, meta}, pages en parallèle.
        `first` : page 1 déjà téléchargée, pour ne pas la re-demander."""
        if first is None:
            first = self.get(path, page=1, limit=limit, **params)
        records = list(first.get("data") or [])
        total_pages = (first.get("meta") or {}).get("totalPages") or 1
        if total_pages > 1:
            with ThreadPoolExecutor(max_workers=PAGE_WORKERS) as pool:
                payloads = pool.map(
                    lambda p: self.get(path, page=p, limit=limit, **params),
                    range(2, total_pages + 1))
                for payload in payloads:
                    records.extend(payload.get("data") or [])
        return records


class JsonlStore:
    """Append thread-safe vers un fichier JSONL, avec flush par écriture."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._fh = open(path, "a", encoding="utf-8")

    def append(self, records) -> None:
        lines = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        with self._lock:
            self._fh.write(lines)
            self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class State:
    """Checkpoint de progression : entités (dossiers/parlementaires) terminées."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self.done = {"dossiers": [], "parlementaires": []}
        if path.exists():
            self.done.update(json.loads(path.read_text()))

    def is_done(self, kind: str, key: str) -> bool:
        return key in set(self.done[kind])

    def mark_done(self, kind: str, key: str) -> None:
        with self._lock:
            self.done[kind].append(key)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.done))
            tmp.replace(self.path)


def fetch_referentiel(client: ApiClient, path: Path, endpoint: str,
                      paginated: bool = True, limit: int = PAGE_LIMIT, **params) -> list:
    """Télécharge une liste complète (dossiers, parlementaires, groupes)."""
    if path.exists():
        records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]
        log(f"{path.name} : {len(records)} enregistrements (cache réutilisé)")
        return records
    if paginated:
        records = client.get_all_pages(endpoint, limit, **params)
    else:
        records = client.get(endpoint, **params).get("data") or []
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"{path.name} : {len(records)} enregistrements téléchargés")
    return records


def run_phase(client: ApiClient, state: State, kind: str, entities: list,
              key_field: str, worker, workers: int) -> None:
    """Exécute `worker(entity)` en parallèle sur les entités non encore traitées."""
    todo = [e for e in entities if not state.is_done(kind, e[key_field])]
    log(f"Phase {kind} : {len(todo)} à traiter ({len(entities) - len(todo)} déjà faits)")
    if not todo:
        return
    log_every = 25 if len(todo) > 100 else 1
    done_count = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, e): e for e in todo}
        for fut in as_completed(futures):
            entity = futures[fut]
            fut.result()  # propage les erreurs
            state.mark_done(kind, entity[key_field])
            done_count += 1
            if done_count % log_every == 0 or done_count == len(todo):
                rate = done_count / (time.time() - t0)
                eta = (len(todo) - done_count) / rate if rate else 0
                log(f"  {kind}: {done_count}/{len(todo)} "
                    f"({client.requests_done} requêtes, ETA {eta / 60:.0f} min)")


class GroupeResolver:
    """Résout le groupe politique des signataires (par slug) et de l'auteur
    (par parsing du premier nom de `auteurLibelle`)."""

    def __init__(self, parlementaires: list, groupes: list):
        groupe_by_id = {g["id"]: g for g in groupes}
        self.parl_by_slug = {}
        self.parls_by_nom = {}
        self.parls_by_prenom_nom = {}
        for p in parlementaires:
            groupe = groupe_by_id.get(p.get("groupeId"))
            info = {
                "slug": p["slug"],
                "nom": p.get("nom"),
                "prenom": p.get("prenom"),
                "chambre": p.get("chambre"),
                "groupe": groupe.get("nom") if groupe else None,
                "groupeNomComplet": groupe.get("nomComplet") if groupe else None,
                "couleur": groupe.get("couleur") if groupe else None,
                "position": groupe.get("position") if groupe else None,
            }
            self.parl_by_slug[p["slug"]] = info
            self.parls_by_nom.setdefault(normalize(p.get("nom") or ""), []).append(info)
            self.parls_by_prenom_nom.setdefault(
                normalize(f"{p.get('prenom') or ''} {p.get('nom') or ''}"), []).append(info)

        self._name_cache = {}

    def signataire(self, slug: str) -> dict:
        return self.parl_by_slug.get(slug) or {"slug": slug}

    def resolve_name(self, name: str):
        """Résout un nom tel qu'écrit dans auteurLibelle ("M. Molac",
        "Mme Perrine Goulet") vers un parlementaire. None si inconnu ou ambigu."""
        if name in self._name_cache:
            return self._name_cache[name]
        nom = normalize(CIVILITE_RE.sub("", name))
        candidats = (self.parls_by_nom.get(nom)
                     or self.parls_by_prenom_nom.get(nom) or [])
        info = candidats[0] if len(candidats) == 1 else None
        self._name_cache[name] = info
        return info

    def auteur(self, auteur_libelle, signataire_slugs) -> dict:
        """Groupe de l'auteur = 1er nom de auteurLibelle, cherché en priorité
        parmi les signataires de l'amendement, sinon dans tout le référentiel."""
        if not auteur_libelle:
            return {}
        # "Mme A, M. B et M. C" ou "Mme A et Mme B" -> premier nom cité
        premier = re.split(r",| et ", auteur_libelle)[0].strip()
        if normalize(premier).startswith("le gouvernement"):
            return {"nom": "Le Gouvernement", "groupe": "GOUVERNEMENT"}
        # nom seul ("M. Molac") ou prénom + nom en cas d'homonymie ("Mme Perrine Goulet")
        nom = normalize(CIVILITE_RE.sub("", premier))
        candidats = [self.parl_by_slug[s] for s in signataire_slugs
                     if s in self.parl_by_slug
                     and normalize(self.parl_by_slug[s]["nom"] or "") == nom]
        if not candidats:
            candidats = (self.parls_by_nom.get(nom)
                         or self.parls_by_prenom_nom.get(nom) or [])
        groupes = {c["groupe"] for c in candidats}
        if len(candidats) == 1 or len(groupes) == 1 and candidats:
            c = candidats[0]
            return {"nom": premier, "slug": c["slug"] if len(candidats) == 1 else None,
                    "groupe": c["groupe"], "couleur": c["couleur"],
                    "position": c["position"]}
        return {"nom": premier}  # introuvable ou ambigu


class CouvertureTracker:
    """Suivi des amendements déjà téléchargés, pour éviter de re-télécharger
    un parlementaire qui n'apportera rien de nouveau.

    Pour chaque parlementaire on compte les amendements déjà en local qui le
    citent comme signataire (parsé depuis auteurLibelle, dont la liste est
    complète). Si ce compte atteint le total annoncé par l'API pour lui ET que
    sa première page ne contient que des uids connus, on le saute : tous ses
    amendements ont déjà été récupérés via les dossiers ou ses cosignataires."""

    def __init__(self, resolver: GroupeResolver):
        self.resolver = resolver
        self.known_uids = set()
        self.counts = {}
        self._libelle_cache = {}  # les libellés se répètent massivement
        self._lock = threading.Lock()

    def _slugs(self, libelle: str) -> tuple:
        slugs = self._libelle_cache.get(libelle)
        if slugs is None:
            slugs = tuple({info["slug"] for name in split_names(libelle)
                           if (info := self.resolver.resolve_name(name))})
            self._libelle_cache[libelle] = slugs
        return slugs

    def ingest(self, amendements) -> None:
        with self._lock:
            for a in amendements:
                uid = a.get("uid")
                if uid in self.known_uids:
                    continue
                self.known_uids.add(uid)
                for slug in self._slugs(a.get("auteurLibelle") or ""):
                    self.counts[slug] = self.counts.get(slug, 0) + 1

    def deja_couvert(self, slug: str, total: int, premiere_page: list) -> bool:
        with self._lock:
            return (self.counts.get(slug, 0) == total
                    and all(a.get("uid") in self.known_uids for a in premiere_page))

    def preload(self, paths) -> None:
        """Réindexe les fichiers bruts d'une exécution précédente (reprise)."""
        for path in paths:
            if path.exists():
                with open(path, encoding="utf-8") as fh:
                    self.ingest(json.loads(line)["amendement"] for line in fh)


def merge_and_export(out_dir: Path, raw_dossiers: Path, raw_parls: Path,
                     resolver: GroupeResolver) -> dict:
    """Fusionne les deux passes, déduplique par uid, enrichit, exporte JSONL + CSV."""
    log("Fusion et déduplication…")
    amendements = {}

    def iter_jsonl(path):
        """Ignore une éventuelle dernière ligne tronquée (extraction en cours)."""
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    for rec in iter_jsonl(raw_dossiers):
        amdt = rec["amendement"]
        amdt["dossier"] = rec["dossier"]
        amdt["_signataire_slugs"] = []
        amendements[amdt["uid"]] = amdt

    seuls_via_parlementaires = 0
    for rec in iter_jsonl(raw_parls):
        slug = rec["slug"]
        amdt = rec["amendement"]
        existing = amendements.get(amdt["uid"])
        if existing is None:
            amdt["_signataire_slugs"] = [slug]
            amendements[amdt["uid"]] = amdt
            seuls_via_parlementaires += 1
        else:
            # slug déjà vu = reprise après interruption au milieu d'une entité
            if slug not in existing["_signataire_slugs"]:
                existing["_signataire_slugs"].append(slug)
            # la route parlementaire porte des champs absents de la route dossier
            for field in ("legislature", "chambre", "dateSort", "dossier"):
                if existing.get(field) is None and amdt.get(field) is not None:
                    existing[field] = amdt[field]

    for amdt in amendements.values():
        slugs = amdt.pop("_signataire_slugs")
        amdt["auteur"] = resolver.auteur(amdt.get("auteurLibelle"), slugs)
        # signataires = slugs observés via la route parlementaire, complétés par
        # le parsing de auteurLibelle (parlementaires sautés, ex-parlementaires…)
        signataires = [resolver.signataire(s) for s in slugs]
        vus = set(slugs)
        for name in split_names(amdt.get("auteurLibelle") or ""):
            if normalize(name).startswith("le gouvernement"):
                continue
            info = resolver.resolve_name(name)
            if info is None:
                signataires.append({"nom": name})  # inconnu du référentiel ou homonyme
            elif info["slug"] not in vus:
                vus.add(info["slug"])
                signataires.append(info)
        amdt["signataires"] = signataires

    jsonl_path = out_dir / "amendements.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for amdt in amendements.values():
            fh.write(json.dumps(amdt, ensure_ascii=False) + "\n")

    csv_path = out_dir / "amendements.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for amdt in amendements.values():
            dossier = amdt.get("dossier") or {}
            auteur = amdt.get("auteur") or {}
            writer.writerow({
                **amdt,
                "auteur_nom": auteur.get("nom"),
                "auteur_groupe": auteur.get("groupe"),
                "auteur_couleur": auteur.get("couleur"),
                "signataires": "|".join(s.get("slug") or s.get("nom") or ""
                                        for s in amdt["signataires"]),
                "signataires_groupes": "|".join(s.get("groupe") or "?" for s in amdt["signataires"]),
                "dossier_uid": dossier.get("uid"),
                "dossier_titre": dossier.get("titre"),
                "nb_scrutins": len(amdt.get("scrutins") or []),
            })

    log(f"→ {jsonl_path} ({len(amendements)} amendements)")
    log(f"→ {csv_path}")
    return {
        "amendements_uniques": len(amendements),
        "amendements_hors_dossier": seuls_via_parlementaires,
    }


def load_dotenv() -> None:
    """Charge un éventuel fichier .env (KEY=value) sans écraser l'environnement.
    Cherché dans le répertoire courant puis à la racine du repo."""
    candidates = [Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"]
    env_path = next((p for p in candidates if p.exists()), None)
    if env_path is None:
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Extraction de tous les amendements de l'API CLAIR")
    parser.add_argument("--out", default="data", help="répertoire de sortie (défaut: data/)")
    parser.add_argument("--workers", type=int, default=8,
                        help="entités traitées en parallèle (défaut: 8 ; "
                             "chaque entité télécharge ses pages sur "
                             f"{PAGE_WORKERS} connexions)")
    parser.add_argument("--api-key", default=os.environ.get("CLAIR_API_KEY"),
                        help="clé API (défaut: variable d'environnement CLAIR_API_KEY)")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore l'état sauvegardé et repart de zéro")
    parser.add_argument("--max-entities", type=int, default=None,
                        help="(test) ne traiter que N dossiers et N parlementaires")
    args = parser.parse_args()

    if not args.api_key:
        log("ATTENTION : pas de clé API (CLAIR_API_KEY) → rate-limit sévère. "
            "Réduction à 1 worker.")
        args.workers = 1

    out_dir = Path(args.out)
    state_dir = out_dir / "_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    raw_dossiers_path = state_dir / "raw_dossiers.jsonl"
    raw_parls_path = state_dir / "raw_parlementaires.jsonl"

    if args.fresh:
        for p in state_dir.iterdir():
            p.unlink()

    client = ApiClient(args.api_key)
    state = State(state_dir / "state.json")
    started_at = datetime.now(timezone.utc).isoformat()

    stats = client.get("/analytics/stats").get("data", {})
    log(f"Base CLAIR : {stats.get('totalAmendements')} amendements annoncés, "
        f"{stats.get('totalDeputes')} parlementaires")

    dossiers = fetch_referentiel(client, out_dir / "dossiers.jsonl", "/dossiers/")
    parlementaires = fetch_referentiel(client, out_dir / "parlementaires.jsonl",
                                       "/parlementaires/", limit=200, actif="true")
    groupes = fetch_referentiel(client, out_dir / "groupes.jsonl",
                                "/parlementaires/groupes", paginated=False)
    resolver = GroupeResolver(parlementaires, groupes)

    if args.max_entities:
        dossiers = dossiers[: args.max_entities]
        parlementaires = parlementaires[: args.max_entities]

    tracker = CouvertureTracker(resolver)
    t0 = time.time()
    tracker.preload([raw_dossiers_path, raw_parls_path])
    if tracker.known_uids:
        log(f"Reprise : {len(tracker.known_uids)} amendements déjà en local "
            f"(réindexés en {time.time() - t0:.0f}s)")
    sautes = []  # list.append est thread-safe (CPython)

    raw_dossiers = JsonlStore(raw_dossiers_path)
    raw_parls = JsonlStore(raw_parls_path)
    try:
        def fetch_dossier(dossier: dict) -> None:
            amdts = client.get_all_pages(
                f"/dossiers/{dossier['uid']}/amendements", PAGE_LIMIT)
            tracker.ingest(amdts)
            context = {"uid": dossier["uid"], "titre": dossier.get("titre"),
                       "titreCourt": dossier.get("titreCourt")}
            raw_dossiers.append(
                {"dossier": context, "amendement": a} for a in amdts)

        def fetch_parlementaire(parl: dict) -> None:
            path = f"/parlementaires/{parl['slug']}/amendements"
            # sonde : 1 requête pour savoir si tout est déjà en local
            first = client.get(path, page=1, limit=PAGE_LIMIT_PARL)
            meta = first.get("meta") or {}
            if tracker.deja_couvert(parl["slug"], meta.get("total") or 0,
                                    first.get("data") or []):
                sautes.append(parl["slug"])
                return
            amdts = client.get_all_pages(path, PAGE_LIMIT_PARL, first=first)
            tracker.ingest(amdts)
            raw_parls.append(
                {"slug": parl["slug"], "amendement": a} for a in amdts)

        run_phase(client, state, "dossiers", dossiers, "uid",
                  fetch_dossier, args.workers)
        run_phase(client, state, "parlementaires", parlementaires, "slug",
                  fetch_parlementaire, args.workers)
        if sautes:
            log(f"Parlementaires sautés (amendements déjà tous couverts) : {len(sautes)}")
    finally:
        raw_dossiers.close()
        raw_parls.close()

    counts = merge_and_export(out_dir, raw_dossiers_path, raw_parls_path, resolver)

    meta = {
        "extraction_start": started_at,
        "extraction_end": datetime.now(timezone.utc).isoformat(),
        "api": BASE_URL,
        "amendements_annonces_par_api": stats.get("totalAmendements"),
        "dossiers": len(dossiers),
        "parlementaires": len(parlementaires),
        "parlementaires_sautes_deja_couverts": len(sautes),
        "requetes_http": client.requests_done,
        **counts,
    }
    (out_dir / "extraction_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))
    log(f"Terminé. {json.dumps(counts, ensure_ascii=False)}")
    log("Les fichiers de travail dans data/_state/ peuvent être supprimés.")


if __name__ == "__main__":
    main()
