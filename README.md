## Modellierung von Leseverhalten auf Online-Zeitungsportalen anhand von Deep Learning
#### Masterarbeit von Susanna Rücker an der FSU Jena in Zusammenarbeit mit INWT Statistics
#### Gutachter: [Prof. Dr. Udo Hahn](https://julielab.de/Staff/Hahn/), [Dr. Steffen Wagner](https://www.inwt-statistics.com/home.html)

### English abstract

The present study deals with user engagement on German online news articles.
Several implicit feedback measures or Key Performance Indicators (KPIs) – such as pageviews or dwell time – are commonly used as a proxy for measuring user engagement on websites.
After a thorough discussion concerning the use of implicit user feedback and related fields of research, the main focus of this work is building different kinds of models for predicting average user dwell time given only the text of each article. A large corpus (consisting of news articles and their respective KPI measures) was created specifically for this study, part of which (36383 articles from the German daily newspaper [censored]) was then chosen for the prediction task.
The line of models includes several baselines, two of them using Bag-of-Words features, but relies mostly on including the well-known pretrained transformer model BERT in several Deep Learning architectures. All models are evaluated and compared on unseen test data, using various evaluation metrics.
This work deals with the problem of applying BERT to longer documents, given its limitation on input sequence length. Most of the models simply truncate the article
and just use the first part – which turns out to be a valid approach resulting in good predictions of dwell time. However, two of the more complex models take a hierarchical approach, splitting the article in several smaller sections and combining the output of each section.
A further analysis gives insights on the dwell time predictions of two models (one BOW-baseline and one model including BERT), using the tool SHAP for interpreting model predictions.


### abgegebene schriftliche Version der Arbeit (Update am 10.09.21):

* [Repo mit LaTeX-Files (privates Repository)](https://github.com/susannaruecker/thesis)
* [zensierte Version (Namen der Zeitungen sind unkenntlich gemacht)](censored_MA_Ruecker_2021_Modellierung_von_Leseverhalten.pdf)
* abgegebene, unzensierte Version: auf Anfrage [per Mail](mailto:susanna.ruecker@uni-jena.de)


### Disclaimer zum Repository (Update am 10.09.21)
Dieses Working Repository ist alles andere als aufgeräumt, dokumentiert oder öffentlichkeitsfähig... Code ist unverändert, README wurde etwas angepasst und um pdf erweitert.



### Überblick über den Inhalt des Repository:

in `/master_thesis`:

* `/src`: Modell-Architekturen (models.py), allgemein wichtige Helferfunktionen (utils.py), Einlesen der Daten (read_data.py) und solche zu Dataset/Dataloafer etc. (data.py).

* `/experiments`: Skripte für das Training der unterschiedlichen Modelle. Die relevanten liegen allem in `/regression', die anderen Folder enthalten nicht weitergeführte zusätzliche Experimente zur Umformung in ein binäres Klassifikationsproblem, erste Ansätze für Emotionsanalyse. 

* `/notebooks`: (sehr unsystematische) Jupyter Notebooks mit ersten explorativen Modellierungsversuchen, Datensatzinspektion, Zusammenführen der verschiedenen Datensätze, ...

* `/outputs`: (wird nicht von git getrackt) Gespeicherte trainierte Modelle, gespeicherte Features, Predictions, Tensorboard-logdirs

* `/deprecated`: Veraltetes, das aber vielleicht noch interessant sein könnte und daher nicht gelöscht wurde





### Erste Notizen, hilfreiche Links etc. (nicht fortgesetzt)

* [Mein Google Doc mit Notizen](https://docs.google.com/document/d/1bJId1P24eTJRnnxK0hEekf1k6nQvkHHs-Ud2nz3W2VM/edit)

Venelin Valkov (gute Tutorials, klar erklärt)

* [Preprosessing](https://www.youtube.com/watch?v=Osj0Z6rwJB4)

* [Text Classification](https://www.youtube.com/watch?v=8N-nM3QW7O0)

* [Tutorial with text](https://www.curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/)
* [zu SequenceClassification](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)



zum Problem LANGE Texte:

* https://medium.com/@armandj.olivares/using-bert-for-classifying-documents-with-long-texts-5c3e7b04573d
* Transformer-XL: https://arxiv.org/abs/1901.02860
* Longformer: https://arxiv.org/abs/2004.05150


