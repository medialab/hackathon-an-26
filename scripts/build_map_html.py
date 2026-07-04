#!/usr/bin/env python3
"""Injecte un map_data JSON (gzip+base64) dans viz/carte_template.html.

Usage : build_map_html.py <chemin_json> <chemin_sortie.html> [--fragment]

Par défaut la sortie est un HTML autonome (doctype inclus), ouvrable d'un
double-clic. Avec --fragment, produit le fragment sans doctype attendu par la
plateforme d'artifacts.

Contrainte plateforme (constatée) : au-delà de ~5 Mo de HTML, l'artifact met
15 s+ à s'afficher (écran noir) voire ne s'affiche jamais (~9,5 Mo). Pour les
artifacts, utiliser un JSON échantillonné (build_full_map.py 40000).
"""
import base64
import gzip
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

args = [a for a in sys.argv[1:] if not a.startswith("--")]
fragment = "--fragment" in sys.argv
data = Path(args[0]) if args else ROOT / "data" / "embeddings" / "map_data.json"
out = Path(args[1]) if len(args) > 1 else ROOT / "viz" / "carte.html"
if not data.is_absolute():
    data = ROOT / data
if not out.is_absolute():
    out = ROOT / out

tpl = (ROOT / "viz" / "carte_template.html").read_text()
payload = base64.b64encode(gzip.compress(data.read_bytes(), 9)).decode()
html = tpl.replace("__DATA__", payload)
if not fragment:
    html = f'<!doctype html>\n<html>\n<head><meta charset="utf-8"></head>\n{html}\n</html>\n'
out.write_text(html)
print(f"{out} : {out.stat().st_size/1e6:.1f} Mo ({'fragment artifact' if fragment else 'autonome'}, json {data.stat().st_size/1e6:.1f} Mo)")
