# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hackathon AN 2026 project: cross-analysis of French parliamentary activity (Assemblée nationale) and declared lobbying activity. The goal is to use computational methods to link parliamentary initiatives (bills, amendments, citizen petitions) with the activities declared by interest representatives (lobbyists) to the HATVP (Haute Autorité pour la transparence de la vie publique).

The repository currently contains no code — only the challenge description in README.md. Project documentation and data are in French.

## Data Sources

- Amendments (Assemblée nationale)
- Government and member bills (PJL — projets de loi, PPL — propositions de loi)
- Citizen petitions filed with the Assemblée nationale
- HATVP register of declared interest-representative activities: https://www.hatvp.fr/le-repertoire/
  - Open data download: `curl -LO http://www.hatvp.fr/agora/opendata/agora_repertoire_opendata.json`
