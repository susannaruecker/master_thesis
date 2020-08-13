## Analye von Akzeptanz von dpa-Artikel bei Lesern

Masterarbeit in Zusammenarbeit mit INWT Statistics

### Überblick über den Inhalt des Repository

* read_data
  * Daten einlesen, erste minimale Präprozessierung, Parsen von Metadaten (keywords, subject, city, Textlänge, Teaserlänge, ...)
  * Überblick über vorhandene Spalten/Labels/Metadaten
  * mögliche Labels: pageviews, timeOnPage, entrances
  * Problem: es gibt ein paar Artikel mit sehr hohen Zahlen, der Großteil aber im niedrigeren Bereich, wie geht man damit um? Ausreißer?
    * pageviews: größtenteils <50, aber max ist 3000
    * timeOnPage: größtenteils <1500, aber max ist 200000
    * gilt auch bei Textlänge: Großteil bis 500 Tokens, aber ein paar auch länger (längter ca. 1400 Tokens)
    * Verteilungen erinnern an Zipf
* first_trial_BOW
  * Modellierung von pageviews / avgTimeOnPage mit einfachem Bag-of-Words-Ansatz (lemmatisiert, n-Gramme, Ridge Regression)
  * Rumprobieren, welchen Text man für die Features verwendet: text_preprocessed, teaser, titelH1
  * Evaluation an dev-Set: unterschiedlich gut, je nach Wahl von text_base und target...
  * Ausprobieren von Visualisierung mit SHAP
* first_trial_Embs
  * Fast-Text-Embeddings
  * einfacher Ansatz: Mittelwert der Token-Embeddings als Feature, dann Ridge Regression
  * nicht so gute Evaluation
  * nachdenken über: Groß-/Kleinschreibung? Lemmatisieren?
  * außerdem: dauert eher lange, kann man das beschleunigen?
* first_trial_BERT
  * erstes Probieren mit Bert, pretrained ('bert-base-german-cased')
  * Verwendung von BertForSequenceClassification, aber mit num_labels = 1 (entspricht Regression statt Klassifikation)
  * DataSet, DataLoader erstellt
  * Fragen, die man sich stellen muss:
    * welche text_base?
    * wenn ganzer Text: wie mit den vergleichsweise langen Texten umgehen? abschneiden?
    * GPU wäre gut, dauert alles ziemlich lange
    * AdamW, MSE-Loss geeignet?
    * Epochenzahl, batch size
  * Stand: keine Fehler, aber Modell scheint nicht zu lernen!
* verschiedene Files mit Metadaten
  * meta-dict: Bedeutung der Spalten/Variablen
  * meta_file_{}: DataFrames mit Artikel-Ids mit den ihnen zugeordneten keywords, kategorie, ...







### Fortlaufende Notizen, hilfreiche Links etc.

Venelin Valkov (gute Tutorials, klar erklärt)

* Preprosessing: https://www.youtube.com/watch?v=Osj0Z6rwJB4

* Text Classification: https://www.youtube.com/watch?v=8N-nM3QW7O0

* Tutorial with text: https://www.curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/
* zu SequenceClassification: https://mccormickml.com/2019/07/22/BERT-fine-tuning/



zum Problem LANGE Texte:

* https://medium.com/@armandj.olivares/using-bert-for-classifying-documents-with-long-texts-5c3e7b04573d
* Transformer-XL: https://arxiv.org/abs/1901.02860
* Longformer: https://arxiv.org/abs/2004.05150



* Ridge ist gut, das Problem mit 0 einfach händisch postprozessieren (eher dämlich aber naja)
* bei pageview nur teaser nehmen klingt sinnvoll
* was sind polynomiale Features (oder so?)
* nächster Schritt: Embeddings (fasText) nehmen und einfach mitteln und dann mit sklearn weiter
* dann Bert!
* Trick um Regression mit Bert zu machen: "bert for sequence classification" nehmen und num_labels==1 setzen

