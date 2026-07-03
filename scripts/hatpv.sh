# Flatten data into denomination,objet
xan from data/agora_repertoire_opendata.json.gz --root publications | xan select denomination,exercices | xan explode exercices -S -e '_.parse_json()' | xan explode exercice -e '_.parse_json().publicationCourante.activites || {}' | xan transform exercice '_.parse_json().publicationCourante.objet.try()' -r objet | xan search -s objet -N > objets.csv
