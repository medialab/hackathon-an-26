#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Construit le jeu amendement -> organisme HATVP, version RAPPEL MAXIMISÉ (v2).

Réécriture de scripts/build_gold_hatvp.py (repo medialab/hackathon-an-26)
intégrant l'audit de rappel. Même contrat d'entrée/sortie que la v1, plus :

CANAUX DE DÉTECTION (du plus sûr au moins sûr) :
  1. co_redaction    : amorce explicite d'origine externe, regex CUE fortement
                       élargie (« reprend une proposition de », « conformément
                       aux recommandations de », « comme le préconise »,
                       « sur proposition de », « alerté par »...).
  2. org_sujet_actif : organisme SUJET d'un verbe déclaratif en tête de phrase
                       (« La FNSEA propose... », « Le LEEM recommande... »),
                       motif totalement absent de la v1.
  3. mention_directe : dénomination complète, quasi-exacte (fenêtre de tokens),
                       acronyme ou alias (sigle HATVP, alias externe, acronyme
                       généré à partir des initiales, appris via « Nom (SIGLE) »).
  4. candidate       : résolution sous les seuils gold mais au-dessus des seuils
                       candidats -> fichier d'audit au lieu d'être jeté.

AUTRES GAINS DE RAPPEL vs v1 :
  - multi-organismes par mention (« soutenu par le MEDEF et la CPME » -> 2 labels)
  - fenêtre de capture 80 -> 140 caractères, nettoyage des amorces parasites
    (« avec l'appui de », « le concours de »...)
  - STOPTAIL ne coupe plus « pour » que devant un infinitif / « que »
    (préserve « Association pour la ... »)
  - REFLAW assoupli : rejet seulement si la mention EST une référence législative
    (tête de mention ou majorité de tokens), et jamais si la résolution est parfaite
  - acronymes résolus dans TOUS les canaux, insensibles à la variante « Leem »
  - scan aussi du dispositif (amendment_content) en plus de l'exposé sommaire
  - alias externes optionnels (--aliases alias.csv : colonnes alias,denomination)
    pour brancher Wikidata / SIRENE sans réseau

QUALIFICATION (nouveau) :
  - stance : positif_probable / negatif_probable / non_determine, détectée par
    marqueurs d'opposition autour de la mention (« contrairement à ce que
    réclame X » n'est plus confondu avec un soutien). À fiabiliser ensuite par
    une passe LLM (Mistral) sur les snippets.

Sorties (dans --outdir), CSV uniquement, dédupliquées par org_denomination
(un amendement n'apparaît qu'une fois par organisme) :
  - gold_hatvp.csv            : co_redaction + org_sujet_actif (haute confiance)
  - weak_candidates_hatvp.csv : mention_directe (à auditer)
  - candidates_hatvp.csv      : sous les seuils gold (audit rappel)
  - unresolved_actors_hatvp.csv : acteur cité avec attribution mais ABSENT de la
                                  HATVP (France Urbaine, FNCCR...) — pour mapping ultérieur
  - gold_hatvp_meta.json      : stats du run

Usage :
  python3 build_gold_hatvp_v2.py \
      --input amendements_uniques_2025.csv \
      --hatvp data/agora_repertoire_opendata.json.gz \
      --outdir data/gold_v2 [--aliases aliases.csv]

Dépendances : stdlib uniquement ; utilise rapidfuzz si disponible (sinon
repli difflib équivalent à token_set_ratio).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

# --------------------------------------------------------------------------- #
# Fuzzy matching : rapidfuzz si dispo, sinon repli stdlib (même sémantique)
# --------------------------------------------------------------------------- #

try:
    from rapidfuzz import fuzz as _fuzz

    def token_set_ratio(a: str, b: str) -> float:
        return _fuzz.token_set_ratio(a, b)
except ImportError:  # repli pur stdlib
    from difflib import SequenceMatcher

    def token_set_ratio(a: str, b: str) -> float:
        ta, tb = set(a.split()), set(b.split())
        if not ta or not tb:
            return 0.0
        inter = " ".join(sorted(ta & tb))
        sa = (inter + " " + " ".join(sorted(ta - tb))).strip()
        sb = (inter + " " + " ".join(sorted(tb - ta))).strip()

        def r(x: str, y: str) -> float:
            if not x and not y:
                return 0.0
            return SequenceMatcher(None, x, y).ratio() * 100

        return max(r(inter, sa), r(inter, sb), r(sa, sb))

# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #

_WS = re.compile(r"\s+")
_NONWORD = re.compile(r"[^\w]+", re.UNICODE)


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


# Abréviations canonisées des deux côtés (texte ET répertoire)
_SUBST = [("ile de france", "idf")]


def norm(s: str) -> str:
    """Accents retirés, minuscule, ponctuation -> espace, abréviations canonisées."""
    if not s:
        return ""
    s = strip_accents(s)
    s = _NONWORD.sub(" ", s.lower())
    s = _WS.sub(" ", s).strip()
    for a, b in _SUBST:
        s = s.replace(a, b)
    return s


# Tokens trop génériques pour distinguer un organisme (identique v1).
STOP = {
    "france", "francais", "francaise", "national", "nationale", "nationaux",
    "de", "des", "du", "la", "le", "les", "et", "d", "l", "pour", "en", "au", "aux",
    "association", "associations", "federation", "fedration", "syndicat", "syndicale",
    "union", "groupe", "comite", "conseil", "chambre", "chambres", "societe", "ste",
    "sas", "sarl", "sa", "entreprise", "entreprises", "institut", "centre", "office",
    "agence", "reseau", "collectif", "organisation", "mouvement", "ordre", "compagnie",
    "sur", "par", "avec", "sans", "plus", "grand", "grande",
}

# Articles seulement : pour générer les acronymes par initiales (FHF...).
ARTICLES = {"de", "des", "du", "la", "le", "les", "l", "d", "et", "en", "au", "aux", "a", "pour"}

# --------------------------------------------------------------------------- #
# Amorces (CUE) — fortement élargies
# --------------------------------------------------------------------------- #

_DE = r"(?:de\s+|des\s+|du\s+|d['’]\s*)"

# Amorces FORTES -> éligibles gold (co_redaction)
_CUE_STRONG = [
    # participe de co-élaboration + (connecteur)? + avec/par (+ filler)
    r"(?:travaill[ée]s?|[ée]labor[ée]s?|co[- ]?(?:construits?|r[ée]dig[ée]s?|[ée]labor[ée]s?|[ée]crits?)|"
    r"r[ée]dig[ée]s?|pr[ée]par[ée]s?|con[çc]us?|d[ée]velopp[ée]s?)\s+"
    r"(?:en\s+lien\s+|conjointement\s+|en\s+collaboration\s+|en\s+partenariat\s+|"
    r"en\s+concertation\s+|en\s+coordination\s+|[ée]troitement\s+)?(?:avec|par)\s+",
    r"(?:en\s+lien|en\s+collaboration|en\s+partenariat|en\s+concertation|en\s+coordination)\s+avec\s+",
    # participes passifs + par
    r"(?:propos[ée]s?|sugg[ée]r[ée]s?|port[ée]s?|transmis|remont[ée]s?|soutenus?|demand[ée]s?|"
    r"souhait[ée]s?|recommand[ée]s?|pr[ée]conis[ée]s?|d[ée]fendus?|relay[ée]s?|appuy[ée]s?|"
    r"inspir[ée]s?|formul[ée]s?|salu[ée]s?)\s+par\s+",
    r"(?:alert[ée]e?s?|saisie?s?|sollicit[ée]e?s?|interpell[ée]e?s?)\s+par\s+",
    # à l'initiative / à la demande / sur proposition / issu de / à la suite de
    r"(?:[aà]\s+l['’]?\s*initiative|[aà]\s+la\s+demande|[aà]\s+la\s+suggestion|issue?s?|"
    r"sur\s+(?:la\s+)?proposition|[aà]\s+la\s+suite\s+(?:de\s+la\s+demande|des\s+travaux|des\s+[ée]changes))\s+" + _DE,
    # reprend une proposition / les recommandations de
    r"reprend(?:re|ent)?\s+(?:une?\s+|la\s+|les\s+)?"
    r"(?:propositions?|recommandations?|pr[ée]conisations?|demandes?|travaux|amendements?)\s+"
    r"(?:" + _DE + r"|formul[ée]e?s?\s+par\s+|port[ée]e?s?\s+par\s+)",
    # conformément aux recommandations de
    r"conform[ée]ment\s+aux?\s+(?:recommandations?|pr[ée]conisations?|demandes?|attentes?|propositions?)\s+" + _DE,
    # comme le préconise X
    r"comme\s+(?:le\s+|l['’]\s*)?(?:pr[ée]conis(?:e|ent)|recommand(?:e|ent)|soulign(?:e|ent)|"
    r"demand(?:e|ent)|sugg[èe]r(?:e|ent)|propos(?:e|ent)|rappell(?:e|ent)|r[ée]clam(?:e|ent)|souhait(?:e|ent))\s+",
    # répond à une demande de / fait suite à
    r"r[ée]pond(?:re|ent)?\s+[aà]\s+(?:une?\s+|la\s+|aux?\s+)?(?:demandes?|attentes?|pr[ée]occupations?|inqui[ée]tudes?|sollicitations?)\s+" + _DE,
    r"fait\s+suite\s+(?:[aà]\s+la\s+demande\s+de\s+|aux?\s+(?:[ée]changes?|travaux|sollicitations?|demandes?|auditions?)\s+(?:avec\s+|men[ée]s?\s+avec\s+|" + _DE + r"))",
    # en réponse aux ... par/de
    r"en\s+r[ée]ponse\s+aux?\s+[^.,;:!?]{0,50}?(?:exprim[ée]e?s?\s+|formul[ée]e?s?\s+)?par\s+",
    # après consultation de/avec
    r"apr[èe]s\s+(?:consultations?|concertations?|[ée]changes?|discussions?|auditions?)\s+(?:" + _DE + r"|avec\s+)",
    # le fruit d'une concertation avec
    r"le\s+fruit\s+d['’]\s*(?:une?\s+)?(?:concertations?|collaborations?|travail|dialogues?|[ée]changes?)\s+(?:men[ée]e?s?\s+)?avec\s+",
    # s'inspire des travaux de
    r"s['’]\s*inspir(?:e|ent|ant)\s+(?:des\s+travaux\s+|des\s+recommandations?\s+|de\s+la\s+proposition\s+|largement\s+|directement\s+)?" + _DE,
    # traduit la recommandation (n°X) (du rapport) de
    r"traduit\s+(?:une?\s+|la\s+|les\s+)?(?:recommandations?|pr[ée]conisations?|propositions?|demandes?)"
    r"(?:\s+n\s*[°o]?\s*\d+)?(?:\s+du\s+rapport)?\s+" + _DE,
    # avec l'appui / le concours / le soutien de
    r"avec\s+(?:l['’]\s*appui|le\s+concours|le\s+soutien|l['’]\s*aide|l['’]\s*expertise)\s+" + _DE,
]

# Amorces FAIBLES -> plafonnées au tier weak (trop génériques pour du gold)
_CUE_WEAK = [
    r"selon\s+(?:les\s+recommandations?\s+de\s+|les\s+travaux\s+de\s+|les\s+donn[ée]es\s+de\s+)?",
    r"notamment\s+(?:par\s+|avec\s+|" + _DE + r")?",
    r"[aà]\s+l['’]\s*[ée]coute\s+" + _DE,
    r"d['’]\s*apr[èe]s\s+",
    r"aux\s+c[oô]t[ée]s\s+" + _DE,
]

CUE_STRONG = re.compile("|".join(_CUE_STRONG), re.IGNORECASE)
CUE_WEAK = re.compile("|".join(_CUE_WEAK), re.IGNORECASE)

# Fillers en tête de mention capturée (à retirer avant résolution)
LEAD_FILLER = re.compile(
    r"^(?:l['’]\s*appui\s+de\s+|le\s+concours\s+de\s+|le\s+soutien\s+de\s+|les\s+services\s+de\s+|"
    r"les\s+[ée]quipes\s+de\s+|les\s+repr[ée]sentants?\s+de\s+|nombreux(?:es)?\s+|plusieurs\s+|"
    r"celles?\s+de\s+|ceux\s+de\s+|notamment\s+|en\s+particulier\s+)+",
    re.IGNORECASE,
)

# Fin de mention : ponctuation forte, connecteurs ; « pour » seulement devant
# infinitif ou « que » (préserve « Association pour la protection... »).
STOPTAIL = re.compile(
    r"[.,;:!?«»\"()\n]"
    r"|\b(?:qui|que|dont|afin|visant|vise|permet|permettant|concernant)\b"
    r"|\bpour\s+(?=que\b|[a-zàâäéèêëîïôöùûüç]+(?:er|ir|re)\b)"
    r"|\b(?:et\s+de\b|dans\s+l[ea]\b|sur\s+l|il\s|elle\s|ce\s|cet\s)",
    re.IGNORECASE,
)

# Verbes déclaratifs pour le canal « organisme sujet actif ».
SUBJ_VERB = re.compile(
    r"(?:^|[.!?;]\s+|\n)\s*"
    r"(?:ainsi[, ]\s*)?(?:la\s+|le\s+|les\s+|l['’]\s*|une?\s+|plusieurs\s+|de\s+nombreux(?:es)?\s+)?"
    r"([A-ZÀ-Ý][\w'’&.()\- ]{2,90}?)\s+"
    r"(?:nous\s+|le\s+|la\s+|l['’]\s*)?"
    r"(?:propose(?:nt)?|recommande(?:nt)?|demande(?:nt)?|pr[ée]conise(?:nt)?|alerte(?:nt)?|"
    r"r[ée]clame(?:nt)?|souligne(?:nt)?|sugg[èe]re(?:nt)?|appelle(?:nt)?\s+[aà]|plaide(?:nt)?|"
    r"milite(?:nt)?|salue(?:nt)?|soutien(?:t|nent)|souhaite(?:nt)?|estime(?:nt)?|"
    r"consid[èe]re(?:nt)?|rappelle(?:nt)?|recommandent|d[ée]nonce(?:nt)?|s['’]\s*inqui[èe]te(?:nt)?)\b",
)

# Sujets à ignorer dans le canal sujet actif (institutions, méta-discours).
SUBJ_SKIP = re.compile(
    r"^(?:gouvernement|pr[e]sent amendement|amendement|rapporteur|rapporteure|commission|senat|"
    r"assemblee|assemblee nationale|france|auteur|auteurs|deputes?|senateurs?|groupe|loi|texte|"
    r"conseil constitutionnel|conseil d etat|cour des comptes|etat|article)\b"
)

# Motif « Nom Développé (SIGLE) » -> apprentissage d'alias local au run.
PAREN_ACRO = re.compile(r"([A-ZÀ-Ý][\w'’&,.\- ]{3,90}?)\s*\(\s*([A-Z][A-ZÉÈ\-]{1,11})\s*\)")

# --------------------------------------------------------------------------- #
# Références législatives / temporelles (assoupli vs v1)
# --------------------------------------------------------------------------- #

REFLAW_TOKENS = {
    "loi", "lois", "article", "articles", "alinea", "decret", "ordonnance", "directive",
    "reglement", "amendement", "amendements", "code", "traite", "arrete", "jurisprudence",
    "janvier", "fevrier", "mars", "avril", "mai", "juin", "juillet", "aout", "septembre",
    "octobre", "novembre", "decembre",
    "gouvernement", "senat", "assemblee", "ailleurs", "consequent", "exemple", "nombreux",
}
# Tête de mention typiquement législative (hors « convention/proposition/commission »
# qui figurent dans des noms d'organismes légitimes).
REFLAW_HEAD = re.compile(
    r"^(?:la |le |l |les )?(?:loi|article|alinea|decret|code|ordonnance|directive|"
    r"reglement|arrete|traite|jurisprudence|amendement)s?\b"
)


def looks_like_lawref(m_norm: str) -> bool:
    toks = m_norm.split()
    if not toks:
        return True
    if REFLAW_HEAD.match(m_norm):
        return True
    bad = sum(t in REFLAW_TOKENS for t in toks)
    return bad / len(toks) >= 0.5


# --------------------------------------------------------------------------- #
# Extraction de l'ACTEUR nommé (conservé même s'il n'est PAS à la HATVP)
# --------------------------------------------------------------------------- #

# On coupe une énumération "X et Y" seulement si la partie droite est une NOUVELLE
# entité (déterminant la/le/les/l' ou majuscule), pas une suite génitive
# ("Chambre de Commerce et d'Industrie", "Métiers et de l'Artisanat" -> pas coupé).
_ACTOR_SPLIT = re.compile(r"\s*,\s*|\s+ainsi que\s+|\s*&\s*|\s+et\s+(?=(?:l[ae]s?\s|l['’]|[A-ZÉÈÀ]))",
                          re.IGNORECASE)
_LEAD_DET = re.compile(r"^(?:les|le|la|l['’]|du|des|de la|de l['’]|de|d['’]|au|aux|un|une|"
                       r"notre|nos|leur|leurs|son|sa|ses)\s+", re.IGNORECASE)
# Mots capitalisés mais non-distinctifs (déterminants/pronoms/tête de phrase).
_ACTOR_STOP = {
    "cette", "cet", "ces", "ce", "il", "elle", "ils", "elles", "cela", "celui", "celle",
    "celles", "ceux", "leur", "leurs", "notre", "nos", "votre", "vos", "mon", "ma", "mes",
    "son", "sa", "ses", "chaque", "tout", "toute", "tous", "toutes", "aucun", "aucune",
    "plusieurs", "certain", "certaine", "certains", "certaines", "le", "la", "les", "un",
    "une", "des", "du", "de", "au", "aux", "ledit", "ladite", "meme", "memes", "en",
}


def clean_actor(s: str) -> str:
    s = LEAD_FILLER.sub("", s.strip(" \t'’\"«»().,;:-–\xa0"))
    prev = None
    while s and s != prev:               # retire les déterminants de tête (empilés)
        prev = s
        s = _LEAD_DET.sub("", s).strip()
    return s


# Tokens de méta-discours (l'énoncé parle de l'amendement, pas d'un organisme).
_META_TOKENS = {"amendement", "amendements", "proposition", "present", "presente",
                "dispositif", "mesure", "disposition", "objet", "sous", "redaction",
                "groupe", "groupes"}
# Connecteurs de tête -> capture parasite en début de phrase.
_LEAD_JUNK = {"comme", "tandis", "ainsi", "cependant", "or", "donc", "car", "puisque",
              "pourquoi", "c", "par", "selon", "enfin", "toutefois", "neanmoins", "aussi",
              "ici", "en", "il", "elle", "ce", "cet", "cette"}


def good_actor(s: str) -> bool:
    """Ressemble à un nom d'organisme (pas une réf. de loi, un mot courant, du méta-discours)."""
    if not s:
        return False
    ns = norm(s)
    if len(ns) < 3 or ns in BLOCKLIST_NORM or looks_like_lawref(ns):
        return False
    toks = ns.split()
    if toks[0] in _LEAD_JUNK or any(t in _META_TOKENS for t in toks):
        return False
    # exige au moins un token "propre" : capitalisé et pas un déterminant/pronom.
    return any(t[:1].isupper() and norm(t) not in _ACTOR_STOP for t in s.split())


def split_actors(raw: str):
    """Acteurs nettoyés d'une capture (gère les énumérations, dédoublonne)."""
    parts = [clean_actor(x) for x in _ACTOR_SPLIT.split(raw)]
    cands = [p for p in parts if p] if sum(bool(p) for p in parts) > 1 else [clean_actor(raw)]
    out, seen = [], set()
    for a in cands:
        na = norm(a)
        if a and na not in seen and good_actor(a):
            seen.add(na)
            out.append(a)
    return out


# Dénominations HATVP = mots courants -> exclues de la mention directe (identique v1).
BLOCKLIST_NORM = {
    "realites", "action publique", "seance publique", "equipe", "l equipe", "printemps",
    "interet a agir", "avril", "avenir", "horizon", "horizons", "dialogue", "transition",
    "ambition", "esperance", "liberte", "egalite", "solidarite", "renaissance", "convergence",
    "perspectives", "engagement", "territoires", "generation", "generations",
    "enquete", "etude", "etudes", "audition", "auditions", "consultation", "rapport",
    "mission", "observatoire", "sondage", "concertation", "reflexion", "experience",
}

# Mots français courants : un acronyme identique ne matche qu'en casse exacte.
COMMON_WORDS = {
    "aides", "cause", "canal", "salon", "objet", "pacte", "campagne", "demain", "agir",
    "forum", "idee", "idees", "choix", "place", "terre", "ville", "vie", "air", "eau",
}

# Mots-outils français : un alias identique (ex. sigle « SA ») matcherait ces mots
# partout (résolution insensible à la casse) -> on ne les indexe JAMAIS comme alias.
FR_FUNCTION_WORDS = {
    "sa", "se", "ses", "son", "sur", "par", "sous", "dans", "avec", "sans", "pour",
    "aux", "au", "car", "mais", "donc", "que", "qui", "les", "des", "une", "est",
    "ont", "mes", "tes", "nos", "vos", "leur", "lui", "eux", "elle", "ils", "nous",
    "vous", "tout", "tous", "meme", "cela", "ceux", "via", "dit", "tel", "tels",
    "ainsi", "cet", "cette", "ces", "celle", "notre", "votre", "ne", "pas", "plus",
    "ou", "ni", "en", "de", "du", "la", "le", "et",
}

# Marqueurs d'opposition -> stance negatif_probable (fenêtre autour de la mention).
NEG_MARKERS = [
    "contrairement a", "contre l avis", "a l inverse de", "a rebours", "malgre l opposition",
    "malgre les demandes", "s oppose", "s opposent", "opposition de", "denonce", "denoncent",
    "rejette", "rejettent", "refuse", "refusent", "conteste", "contestent", "critique",
    "critiquent", "deplore", "deplorent", "au detriment", "sous la pression", "en depit",
    "contre la volonte", "balaye les", "ignore les",
]


def detect_stance(text: str, pos: int, evidence: str) -> str:
    """Polarité probable de la mention : fenêtre ±90 caractères autour de la position."""
    window = norm(text[max(0, pos - 90):pos + 90])
    for marker in NEG_MARKERS:
        if marker in window:
            return "negatif_probable"
    if evidence in ("co_redaction", "org_sujet_actif"):
        return "positif_probable"
    return "non_determine"

# --------------------------------------------------------------------------- #
# Répertoire HATVP -> index de résolution (+ alias, acronymes générés)
# --------------------------------------------------------------------------- #

def _find_all(node, key, out):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key and v:
                out.append(v)
            _find_all(v, key, out)
    elif isinstance(node, list):
        for v in node:
            _find_all(v, key, out)


def _initials(denom_norm: str) -> str:
    """Initiales des tokens non-articles : FEDERATION HOSPITALIERE DE FRANCE -> FHF."""
    toks = [t for t in denom_norm.split() if t not in ARTICLES]
    return "".join(t[0] for t in toks).upper()


def load_hatvp(path: Path, aliases_path: Path | None = None):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        pubs = json.load(f)["publications"]

    orgs = []
    for p in pubs:
        denom = (p.get("denomination") or "").strip()
        if not denom:
            continue
        dom, sec = [], []
        _find_all(p.get("activites"), "domainesIntervention", dom)
        _find_all(p.get("activites"), "listSecteursActivites", sec)
        domaines = sorted({d for lst in dom for d in (lst if isinstance(lst, list) else [lst])
                           if isinstance(d, str)})
        secteurs = sorted({s.get("label") for lst in sec for s in (lst if isinstance(lst, list) else [lst])
                           if isinstance(s, dict) and s.get("label")})
        # Champs alias possibles du JSON HATVP (présents ou non selon les versions)
        extra_aliases = []
        for key in ("sigle", "nomUsage", "denominationUsuelle", "ancienNom"):
            v = p.get(key)
            if isinstance(v, str) and v.strip():
                extra_aliases.append(v.strip())
        orgs.append({
            "denomination": denom,
            "siren": p.get("identifiantNational") or "",
            "categorie": (p.get("categorieOrganisation") or {}).get("categorie") or "",
            "domaines": domaines,
            "secteurs": secteurs,
            "norm": norm(denom),
            "aliases": extra_aliases,
        })

    tok_index = defaultdict(set)          # token distinctif -> {org_idx}
    alias_index = {}                      # alias NORMALISÉ -> org_idx (uniques seulement)
    alias_raw = {}                        # alias casse d'origine -> org_idx
    _alias_collisions = set()

    def add_alias(alias: str, i: int):
        a_norm = norm(alias)
        if (not a_norm or len(a_norm) < 3 or a_norm in BLOCKLIST_NORM
                or a_norm in FR_FUNCTION_WORDS or a_norm in COMMON_WORDS):
            return
        if a_norm in alias_index and alias_index[a_norm] != i:
            _alias_collisions.add(a_norm)   # ambigu -> retiré (précision)
            return
        alias_index[a_norm] = i
        alias_raw[alias] = i

    for i, o in enumerate(orgs):
        toks = o["norm"].split()
        distinct = [t for t in toks if t not in STOP and len(t) >= 4]
        o["distinct_tokens"] = distinct or [t for t in toks if len(t) >= 3]
        for t in o["distinct_tokens"]:
            tok_index[t].add(i)
        d = o["denomination"].strip()
        # 1) dénomination courte tout-en-majuscules = acronyme déclaré (v1, élargi à 10)
        if d.isupper() and 3 <= len(d.replace(" ", "")) <= 10 and " " not in d:
            add_alias(d, i)
        # 2) champs alias du JSON (sigle, nomUsage...)
        for a in o["aliases"]:
            add_alias(a, i)
        # 3) acronyme généré par initiales (FHF, CNB...), seulement 3-8 lettres
        ini = _initials(o["norm"])
        if 3 <= len(ini) <= 8:
            add_alias(ini, i)

    # 4) alias externes (Wikidata / SIRENE / manuel) : CSV alias,denomination
    if aliases_path and aliases_path.exists():
        by_norm = {o["norm"]: i for i, o in enumerate(orgs)}
        by_siren = {o["siren"]: i for i, o in enumerate(orgs) if o["siren"]}
        with open(aliases_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                target = (row.get("denomination") or row.get("siren") or "").strip()
                alias = (row.get("alias") or "").strip()
                i = by_siren.get(target) or by_norm.get(norm(target))
                if alias and i is not None:
                    add_alias(alias, i)

    for a_norm in _alias_collisions:
        alias_index.pop(a_norm, None)

    norm_names = [o["norm"] for o in orgs]
    return orgs, tok_index, alias_index, alias_raw, norm_names

# --------------------------------------------------------------------------- #
# Résolution d'une mention -> organismes HATVP (multi-org, multi-niveaux)
# --------------------------------------------------------------------------- #

def resolve_mention(mention: str, orgs, tok_index, alias_index, norm_names,
                    min_score, min_cover, cand_score, cand_cover):
    """Mention capturée -> TOUS les organismes plausibles (pas seulement le meilleur).

    Retourne [(org_idx, score, cover, match_type, tier)] avec
    tier ∈ {"full", "cand"} selon les seuils atteints.
    """
    mention = LEAD_FILLER.sub("", mention.strip(" \t'’\"«»-–"))
    m = norm(mention)
    if len(m) < 2:
        return []
    m_tokens = set(m.split())

    results = {}

    # a) alias / acronymes présents dans la mention (insensible casse via norm)
    for t in m_tokens:
        i = alias_index.get(t)
        if i is not None:
            results[i] = (100.0, 1.0, "acronyme", "full")

    # b) résolution floue multi-org. On ne rassemble des candidats que via les
    # tokens DISCRIMINANTS : ni trop courts, ni trop fréquents (un token présent
    # dans des centaines d'organismes, ex. "france", ne discrimine rien et coûte cher).
    lawref = looks_like_lawref(m)
    cand = set()
    for t in m_tokens:
        if len(t) >= 4:
            bucket = tok_index.get(t)
            if bucket and len(bucket) <= 200:
                cand |= bucket
    for i in cand:
        if norm_names[i] in BLOCKLIST_NORM or i in results:
            continue
        dt = orgs[i]["distinct_tokens"]
        cover = sum(t in m_tokens for t in dt) / len(dt) if dt else 0.0
        if cover < cand_cover:
            continue
        s = token_set_ratio(m, norm_names[i])
        if s < cand_score:
            continue
        # garde-fou référence législative : rejet sauf résolution parfaite
        if lawref and not (cover == 1.0 and s >= 98):
            continue
        tier = "full" if (s >= min_score and cover >= min_cover) else "cand"
        results[i] = (s, cover, "cue+fuzzy", tier)

    return [(i, s, c, mt, tier) for i, (s, c, mt, tier) in results.items()]


_WORD_RE = re.compile(r"[0-9A-Za-zÀ-ÿ'’&\-]{2,}")


def scan_direct(txt_norm: str, txt_raw: str, orgs, tok_index, alias_lookup, alias_multi,
                min_len, quasi_score):
    """Mentions directes : nom complet exact, quasi-exact (fenêtre de tokens),
    acronyme/alias dans le texte brut. Retourne {org_idx: (match_type, score, pos)}."""
    hits = {}
    padded = f" {txt_norm} "
    text_tokens = txt_norm.split()
    tok_pos = defaultdict(list)
    for pos, t in enumerate(text_tokens):
        tok_pos[t].append(pos)

    # candidats via tokens discriminants seulement (ni trop courts ni trop fréquents)
    cand = set()
    for t in set(text_tokens):
        bucket = tok_index.get(t)
        if bucket and len(bucket) <= 200:
            cand |= bucket

    for i in cand:
        o = orgs[i]
        name = o["norm"]
        if name in BLOCKLIST_NORM or not o["distinct_tokens"]:
            continue
        # 1) substring exact (v1)
        if len(name) >= min_len and f" {name} " in padded:
            hits[i] = ("nom_complet", 100.0, txt_norm.find(name))
            continue
        # 2) quasi-exact : tous les tokens distinctifs présents dans une fenêtre serrée
        dt = o["distinct_tokens"]
        if len(dt) >= 2 and all(t in tok_pos for t in dt):
            positions = sorted(p for t in dt for p in tok_pos[t])
            n_org = len(name.split())
            window = n_org + 3
            # plus petite fenêtre contenant au moins un exemplaire de chaque token
            best = None
            for start_idx in range(len(positions)):
                seen, end_idx = set(), start_idx
                for j in range(start_idx, len(positions)):
                    seen.add(text_tokens[positions[j]])
                    end_idx = j
                    if all(t in seen for t in dt):
                        break
                if all(t in seen for t in dt):
                    span = positions[end_idx] - positions[start_idx]
                    if best is None or span < best[0]:
                        best = (span, positions[start_idx], positions[end_idx])
            if best and best[0] <= window:
                w_txt = " ".join(text_tokens[best[1]:best[2] + 1])
                s = token_set_ratio(w_txt, name)
                if s >= quasi_score:
                    hits[i] = ("quasi_exact", s, txt_norm.find(text_tokens[best[1]]))

    # 3) acronymes / alias : on parcourt les TOKENS du texte (O(tokens)) et on les
    # cherche dans la table d'alias pré-calculée — au lieu de tester ~4000 regex/ligne.
    for tok in set(_WORD_RE.findall(txt_raw)):
        i = alias_lookup.get(tok)
        if i is not None and i not in hits:
            hits[i] = ("acronyme", 100.0, txt_raw.find(tok))
    for a_norm, i in alias_multi:              # alias multi-mots (peu nombreux)
        if i not in hits and f" {a_norm} " in padded:
            hits[i] = ("acronyme", 100.0, txt_norm.find(a_norm))
    return hits

# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

EV_RANK = {"co_redaction": 3, "org_sujet_actif": 2, "mention_directe": 1, "candidate": 0}

HEADER = [
    "row_id", "date", "dossier", "number", "author_name", "author_group",
    "org_denomination", "org_categorie", "org_domaines",
    "evidence", "match_type", "score", "cover", "stance", "source_field", "snippet",
]

# Acteur cité avec attribution mais NON rattaché à la HATVP (à mapper plus tard).
UNRES_HEADER = [
    "row_id", "date", "dossier", "number", "author_name", "author_group",
    "acteur_mention", "nb_acteurs", "evidence", "stance", "source_field", "snippet",
]


def scan_text(text: str, source_field: str, ctx, args):
    """Scanne un champ texte, retourne (matches, unresolved)."""
    orgs, tok_index, alias_index, alias_lookup, alias_multi, norm_names, learned, rcache = ctx
    matches = {}

    def add(i, evidence, match_type, score, cover, pos, tier="full"):
        ev = evidence if tier == "full" else "candidate"
        snip = text[max(0, pos - 30):pos + 130].replace("\n", " ").strip()[:160]
        stance = detect_stance(text, pos, evidence)
        prev = matches.get(i)
        if prev is None or (EV_RANK[ev], score) > (EV_RANK[prev["evidence"]], prev["score"]):
            matches[i] = {"evidence": ev, "match_type": match_type, "score": round(score, 1),
                          "cover": round(cover, 2), "stance": stance,
                          "source_field": source_field, "snippet": snip}

    # Acteurs cités avec attribution mais NON rattachés à la HATVP (on les garde).
    unresolved = {}

    def add_unres(actor, evidence, pos):
        na = norm(actor)
        prev = unresolved.get(na)
        if prev is not None and EV_RANK[evidence] <= EV_RANK[prev["evidence"]]:
            return
        snip = text[max(0, pos - 30):pos + 130].replace("\n", " ").strip()[:160]
        unresolved[na] = {"acteur_mention": actor, "evidence": evidence,
                          "stance": detect_stance(text, pos, evidence),
                          "source_field": source_field, "snippet": snip}

    def R(mention):                       # résolution mémoïsée (mêmes noms très répétés)
        k = norm(mention)
        v = rcache.get(k)
        if v is None:
            v = resolve_mention(mention, orgs, tok_index, alias_index_all, norm_names,
                                args.min_score, args.min_cover, args.cand_score, args.cand_cover)
            rcache[k] = v
        return v

    def unresolved_actors(mention, evidence, pos, whole_res):
        # rien à faire si mention simple déjà rattachée à la HATVP
        if not _ACTOR_SPLIT.search(mention) and any(t == "full" for *_, t in whole_res):
            return
        for actor in split_actors(mention):
            if not any(t == "full" for *_, t in R(actor)):   # aucun rattachement fiable
                add_unres(actor, evidence, pos)

    t_norm = norm(text)

    # 0) apprentissage local « Nom Développé (SIGLE) »
    for mm in PAREN_ACRO.finditer(text):
        long_name, sigle = mm.group(1), mm.group(2)
        res = resolve_mention(long_name, orgs, tok_index, alias_index, norm_names,
                              args.min_score, args.min_cover, args.cand_score, args.cand_cover)
        for i, s, c, _, tier in res:
            if tier == "full":
                learned.setdefault(norm(sigle), i)
                alias_lookup.setdefault(sigle, i)          # sigle réutilisable (run global)
                alias_lookup.setdefault(sigle.upper(), i)

    # alias appris (run local) fusionnés pour la résolution floue
    alias_index_all = dict(alias_index)
    alias_index_all.update(learned)

    # 1) amorces fortes -> co_redaction
    for cue_re, cue_gold in ((CUE_STRONG, True), (CUE_WEAK, False)):
        for m in cue_re.finditer(text):
            tail = text[m.end():m.end() + args.window]
            cut = STOPTAIL.search(tail)
            mention = (tail[:cut.start()] if cut else tail)
            res = R(mention)
            for i, s, c, mt, tier in res:
                evidence = "co_redaction" if cue_gold else "mention_directe"
                if not cue_gold:
                    tier = "full" if tier == "full" else "cand"
                add(i, evidence, f"cue+{mt}" if mt != "cue+fuzzy" else mt, s, c, m.start(), tier)
            # acteur explicitement co-rédacteur mais absent de la HATVP -> conservé
            if cue_gold:
                unresolved_actors(mention, "co_redaction", m.start(), res)

    # 2) organisme sujet actif -> org_sujet_actif
    for m in SUBJ_VERB.finditer(text):
        subj = m.group(1)
        if SUBJ_SKIP.match(norm(subj)):
            continue
        res = R(subj)
        for i, s, c, mt, tier in res:
            add(i, "org_sujet_actif", f"sujet+{mt}", s, c, m.start(1), tier)
        if not any(t == "full" for *_, t in res):   # sujet actif non rattaché -> conservé
            unresolved_actors(subj, "org_sujet_actif", m.start(1), res)

    # 3) mentions directes (n'écrasent pas un canal plus fort)
    for i, (mt, s, pos) in scan_direct(t_norm, text, orgs, tok_index, alias_lookup,
                                       alias_multi, args.min_len, args.quasi_score).items():
        # position approximative dans le texte brut pour le snippet/stance
        pos_raw = max(0, min(len(text) - 1, pos if mt == "acronyme" else
                             int(pos / max(1, len(t_norm)) * len(text))))
        add(i, "mention_directe", mt, s, 1.0, pos_raw)

    return matches, unresolved


def collapse_entity_fanout(entries, orgs, norm_names):
    """Réduit l'éclatement d'une même entité HATVP sur un amendement.

    entries : liste de (org_idx, lab). Deux réductions :
      1. organismes au MÊME jeu de tokens distinctifs (national + comités
         départementaux « Ligue contre le cancer », ou même dénomination à
         plusieurs SIREN) -> on garde la dénomination canonique (la plus courte)
         avec le meilleur label du groupe ;
      2. organismes dont les tokens distinctifs ne sont qu'un SUR-ENSEMBLE d'un
         autre gardé (« Jeunes Agriculteurs de la Vienne » vs « Jeunes
         Agriculteurs ») -> écartés.
    """
    groups = defaultdict(list)
    for i, lab in entries:
        groups[frozenset(orgs[i]["distinct_tokens"])].append((i, lab))
    kept = {}
    for grp in groups.values():
        ci = min(grp, key=lambda il: (len(norm_names[il[0]].split()), len(norm_names[il[0]])))[0]
        kept[ci] = max(grp, key=lambda il: (EV_RANK[il[1]["evidence"]], il[1]["score"]))[1]

    kept_sets = [frozenset(orgs[i]["distinct_tokens"]) for i in kept if orgs[i]["distinct_tokens"]]
    final = {}
    for i, lab in kept.items():
        ts = set(orgs[i]["distinct_tokens"])
        if ts and any(fs < ts for fs in kept_sets):    # sur-ensemble d'un autre gardé
            continue
        final[i] = lab
    return final


def run(args):
    t0 = time.time()
    aliases_path = Path(args.aliases) if args.aliases else None
    orgs, tok_index, alias_index, alias_raw, norm_names = load_hatvp(Path(args.hatvp), aliases_path)
    print(f"[HATVP] {len(orgs)} organisations | {len(alias_index)} alias/acronymes indexés",
          file=sys.stderr)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = {}                              # CSV uniquement (pas de jsonl)
    for tier, stem in (("gold", "gold_hatvp"), ("weak", "weak_candidates_hatvp"),
                       ("cand", "candidates_hatvp")):
        fc = (outdir / f"{stem}.csv").open("w", encoding="utf-8", newline="")
        w = csv.writer(fc)
        w.writerow(HEADER)
        files[tier] = (fc, w)
    # 4e sortie : acteurs cités mais non rattachés à la HATVP (schéma dédié)
    ufc = (outdir / "unresolved_actors_hatvp.csv").open("w", encoding="utf-8", newline="")
    uw = csv.writer(ufc)
    uw.writerow(UNRES_HEADER)
    files["unres"] = (ufc, uw)

    n = n_with_text = 0
    n_labels = defaultdict(int)
    rows_by_tier = defaultdict(set)
    org_counter = defaultdict(lambda: defaultdict(int))
    ev_counter = defaultdict(int)
    stance_counter = defaultdict(int)
    unres_counter = defaultdict(int)
    learned = {}

    # Table d'alias inversée (calculée UNE fois) : token -> org, pour un scan direct
    # en O(tokens du texte) au lieu de ~4000 regex par ligne.
    alias_lookup, alias_multi = {}, []
    for alias, i in alias_raw.items():
        a = alias.strip()
        na = norm(a)
        if not na or na in BLOCKLIST_NORM:
            continue
        if " " in a:
            alias_multi.append((na, i))
            continue
        alias_lookup.setdefault(a, i)
        if len(a) >= 4 and a.isupper() and na not in COMMON_WORDS:
            alias_lookup.setdefault(a.capitalize(), i)

    ctx = (orgs, tok_index, alias_index, alias_lookup, alias_multi, norm_names, learned, {})

    with open(args.input, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_content = args.content_col and args.content_col in (reader.fieldnames or [])
        for row_id, r in enumerate(reader):
            n += 1
            fields = [(args.text_col, (r.get(args.text_col) or "").strip())]
            if has_content and not args.no_content:
                fields.append((args.content_col, (r.get(args.content_col) or "").strip()))
            if not any(t for _, t in fields):
                continue
            n_with_text += 1

            merged = {}          # org_idx -> meilleur label (tous canaux)
            merged_unres = {}
            for source_field, text in fields:
                if not text:
                    continue
                m_matches, m_unres = scan_text(text, source_field, ctx, args)
                for i, lab in m_matches.items():
                    prev = merged.get(i)
                    if prev is None or (EV_RANK[lab["evidence"]], lab["score"]) > \
                                       (EV_RANK[prev["evidence"]], prev["score"]):
                        merged[i] = lab
                for na, urec in m_unres.items():
                    prev = merged_unres.get(na)
                    if prev is None or EV_RANK[urec["evidence"]] > EV_RANK[prev["evidence"]]:
                        merged_unres[na] = urec

            # Réduit l'éclatement d'une même entité (SIREN multiples, comités
            # départementaux, variantes régionales) -> une dénomination par amendement.
            final = collapse_entity_fanout(list(merged.items()), orgs, norm_names)
            resolved_norms = {norm(orgs[i]["denomination"]) for i in final}

            # acteurs cités hors HATVP -> UNE ligne par amendement (acteurs joints)
            actors = [u for na, u in merged_unres.items() if na not in resolved_norms]
            if actors:
                actors.sort(key=lambda u: EV_RANK[u["evidence"]], reverse=True)
                n_labels["unres"] += 1
                rows_by_tier["unres"].add(row_id)
                for u in actors:
                    unres_counter[u["acteur_mention"]] += 1
                lead = actors[0]
                stance_counter[lead["stance"]] += 1
                _, uw_ = files["unres"]
                uw_.writerow([row_id, r.get("date"), r.get("dossier"), r.get("number"),
                              r.get("author_name"), r.get("author_group"),
                              " | ".join(u["acteur_mention"] for u in actors), len(actors),
                              lead["evidence"], lead["stance"], lead["source_field"],
                              lead["snippet"]])

            # UN SEUL organisme par amendement : le meilleur label, tous canaux
            # confondus (evidence puis score). Deux organismes pour un amendement
            # n'ont pas de sens ici -> on ne garde que le plus fiable.
            if final:
                i, lab = max(final.items(),
                             key=lambda il: (EV_RANK[il[1]["evidence"]], il[1]["score"]))
                o = orgs[i]
                ev = lab["evidence"]
                tier = ("gold" if ev in ("co_redaction", "org_sujet_actif")
                        else "weak" if ev == "mention_directe" else "cand")
                n_labels[tier] += 1
                org_counter[tier][o["denomination"]] += 1
                ev_counter[ev] += 1
                stance_counter[lab["stance"]] += 1
                rows_by_tier[tier].add(row_id)
                _, w = files[tier]
                w.writerow([
                    row_id, r.get("date"), r.get("dossier"), r.get("number"),
                    r.get("author_name"), r.get("author_group"),
                    o["denomination"], o["categorie"], " | ".join(o["domaines"]),
                    lab["evidence"], lab["match_type"], lab["score"], lab["cover"],
                    lab["stance"], lab["source_field"], lab["snippet"],
                ])

    for fc, _ in files.values():
        fc.close()

    def tier_meta(tier):
        return {
            "amendements_labellises": len(rows_by_tier[tier]),
            "labels": n_labels[tier],
            "organismes_distincts": len(org_counter[tier]),
            "top_organismes": sorted(org_counter[tier].items(), key=lambda x: -x[1])[:30],
        }

    meta = {
        "version": "v2-recall",
        "input": str(args.input),
        "amendements_total": n,
        "amendements_avec_texte": n_with_text,
        "gold": tier_meta("gold"),
        "weak": tier_meta("weak"),
        "candidates": tier_meta("cand"),
        "unresolved": {
            "amendements": len(rows_by_tier["unres"]),
            "labels": n_labels["unres"],
            "acteurs_distincts": len(unres_counter),
            "top_acteurs": sorted(unres_counter.items(), key=lambda x: -x[1])[:40],
        },
        "par_evidence": dict(ev_counter),
        "par_stance": dict(stance_counter),
        "alias_appris_en_cours_de_run": {k: orgs[i]["denomination"] for k, i in learned.items()},
        "params": {"text_col": args.text_col, "content_col": args.content_col if not args.no_content else None,
                   "min_score": args.min_score, "min_cover": args.min_cover,
                   "cand_score": args.cand_score, "cand_cover": args.cand_cover,
                   "min_len": args.min_len, "quasi_score": args.quasi_score,
                   "window": args.window, "aliases_file": args.aliases},
        "duree_s": round(time.time() - t0, 1),
    }
    (outdir / "gold_hatvp_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    for tier, label in (("gold", "co-rédaction + sujet actif"),
                        ("weak", "mention directe, à auditer"),
                        ("cand", "sous seuils, audit rappel")):
        print(f"[{tier.upper():4}] {len(rows_by_tier[tier])} amendements / {n_labels[tier]} labels / "
              f"{len(org_counter[tier])} organismes  ({label})", file=sys.stderr)
    print(f"[UNRE] {len(rows_by_tier['unres'])} amendements / {n_labels['unres']} labels / "
          f"{len(unres_counter)} acteurs distincts  (cités, hors HATVP — à mapper)", file=sys.stderr)
    print(f"       stances : {dict(stance_counter)}", file=sys.stderr)
    print(f"       sur {n_with_text} amendements avec texte | {meta['duree_s']}s", file=sys.stderr)
    return meta


def main():
    p = argparse.ArgumentParser(
        description="Jeu amendement -> organisme HATVP, rappel maximisé (v2).")
    p.add_argument("--input", required=True, help="CSV nettoyé des amendements.")
    p.add_argument("--text-col", default="amendment_summary",
                   help="Colonne de la motivation / exposé sommaire.")
    p.add_argument("--content-col", default="amendment_content",
                   help="Colonne du dispositif, scannée aussi si présente.")
    p.add_argument("--no-content", action="store_true",
                   help="Ne pas scanner le dispositif (comportement v1).")
    p.add_argument("--hatvp", default="data/agora_repertoire_opendata.json.gz")
    p.add_argument("--aliases", default=None,
                   help="CSV optionnel d'alias externes : colonnes alias,denomination[,siren].")
    p.add_argument("--outdir", default="data/gold")
    p.add_argument("--min-score", type=int, default=90, help="Seuil gold de résolution floue.")
    p.add_argument("--min-cover", type=float, default=0.6,
                   help="Couverture gold des tokens distinctifs.")
    p.add_argument("--cand-score", type=int, default=75, help="Seuil candidat (audit).")
    p.add_argument("--cand-cover", type=float, default=0.4, help="Couverture candidat (audit).")
    p.add_argument("--min-len", type=int, default=8, help="Longueur min. nom complet exact.")
    p.add_argument("--quasi-score", type=int, default=85, help="Seuil du match quasi-exact.")
    p.add_argument("--window", type=int, default=140, help="Fenêtre de capture après amorce.")
    run(p.parse_args())


if __name__ == "__main__":
    main()
