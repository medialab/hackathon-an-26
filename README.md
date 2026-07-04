# Hackathon AN 2026 : Exploration croisée de l’activité parlementaire et celle des représentants d'intérêts

Analyser le rôle des groupes d’intérêts et de plaidoyer dans la décision parlementaire permet de mieux comprendre comment leurs positions, ressources et stratégies d’influence contribuent à orienter la formulation, la négociation et l’adoption des lois, et constituent une composante clé de la vie parlementaire. Autant garantie qu’encadrée par loi, l’influence de la décision publique implique une relation régulière entre représentants d’intérêts et parlementaires sans qu’il existe d’outils ou de ressources permettant de comprendre en détails leurs liens mutuels. 

Ce défi vise à explorer les méthodes computationnelles permettant de rapprocher l’initiative parlementaire des activités déclarées des représentants d’intérêt. Il s’appuiera autant sur les données du Parlement que sur les déclarations d’activités déposées auprès de la Haute Autorité pour la transparence de la vie publique, ainsi que les données des pétitions de l’Assemblée nationale. 

Ces données permettent de capturer les mobilisations et les signaux de la société civile. Elles révèlent à la fois les sujets qui suscitent un large engagement et ceux, moins visibles, mais tout aussi importants, qui émergent. En les analysant, on peut mieux comprendre comment les attentes de la société civile s’alignent ou non avec les projets et propositions de loi discutées au Parlement.

## Documents du défi

- Ensemble des amendements (depuis [medialab/parlement_nlp](https://github.com/medialab/parlement_nlp))
- Ensemble des PJL et PPL
- Pétitions citoyennes déposées à l'Assemblée nationale
- Repertoire des activités déclarées par les représentants d'intérêts (HATVP)
À récupérer ici : https://www.hatvp.fr/le-repertoire/
Par exemple :
```shell
curl -LO http://www.hatvp.fr/agora/opendata/agora_repertoire_opendata.json
```

