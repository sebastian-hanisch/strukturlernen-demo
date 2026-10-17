# Das Diagramm aus Daten lernen: Was PC findet und was offen bleibt – Streamlit-Demo

Dreizehntes Stück der Linie **Kausale Diagramme und Adjustierung** in der „Konzepte“-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning). Nachfolger von [uebertragbarkeit-demo](https://github.com/sebastian-hanisch/uebertragbarkeit-demo): Die Stücke davor setzen das Diagramm **voraus**. Hier soll es aus Messungen entstehen, und die Demo zeigt, bis wohin das gelingt.

## Kernfrage

Der **PC-Algorithmus** testet, welche Größen gegeben welche anderen unabhängig sind, wirft die Kanten dazu hinaus und richtet die Pfeile, die aus **v-Strukturen** (Collidern) folgen. Mehr verraten die Daten nicht: Es bleibt eine **Markov-Äquivalenzklasse** von Diagrammen, und die Wirkung des Faktors ist ein Bereich statt einer Zahl. Und ob die Klasse stimmt, hängt an Annahmen (Treue, Linearität, nichts Verborgenes), die PC nicht prüft.

## Modell

- **Welt:** fünf Größen eines Standortnetzes in einem linearen Gauß-Modell mit Einheitsrauschen: Saison S → Nachfrage Z → hoher Faktor X → Servicegrad Y, dazu Z → Y und Wetter W → Y; Pfeilstärken 0,8 / 0,8 / 0,6 / 0,6 / 0,6, **Wirkung X → Y = 0,6** (Koeffizient von X). Daten aus `SplitMix64`-Strömen (Generator `kausal_generator.py`, wortgleich in allen Stücken der Linie).
- **Wahrer CPDAG:** gerichtet Z → Y, X → Y, W → Y (v-Strukturen am Servicegrad), **ungerichtet S–Z und Z–X**; die Klasse hat **3 Diagramme**.
- **Brüche der Annahmen** (Regler und Presets): fast untreue Kante, ein direkter Pfeil, der einen Umweg aufhebt, eine quadratische Kante, eine verborgene Nachfrage.

## Methodik

- **Test:** Teilkorrelation aus der Inversen der Korrelationsmatrix, Fisher-z (`math.erfc`, kein scipy zur Laufzeit); α als Fehlalarmrate je Test.
- **PC** (`stl_pc.py`, von Grund auf): Skelett nach wachsender Bedingungsmenge (ursprünglich oder **stabil**), v-Strukturen nach den Trennmengen, Vorwissen „Saison und Wetter sind Wurzeln“, Meek-Regeln R1 bis R3.
- **Äquivalenzklasse:** durch Aufzählung der Orientierungen (gleiches Skelett, gleiche v-Strukturen, azyklisch); der **CPDAG** richtet nur Kanten, die in jedem Diagramm der Klasse gleich stehen.
- **Wirkungsbereich** (die Idee von Maathuis/Kalisch/Bühlmann 2009, vereinfacht): je Diagramm der Klasse die Anpassung an die Eltern von X; kleinster bis größter Wert.
- **Treue-Spielraum:** je wahrer Kante die kleinste Teilkorrelation über alle Bedingungsmengen (Bevölkerung).
- **Orakel:** PC mit d-Separation als Test (`dsep_graph.py`, wortgleich aus Stück 2) gibt aus, was mit unendlich vielen Daten käme; Aufzählung der Klasse als zweiter Weg.
- **Baukasten:** vier Größen A bis D, je Paar kein Pfeil oder ein Pfeil in einer Richtung; exakt über alle 543 Diagramme mit vier Knoten.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. Läufe: 300 Wiederholungen je Zelle, feste Zufallsströme.

| Frage | Befund | Test |
|---|---|---|
| Standard-Stichprobe (n = 1 000, α = 0,01, Seed 35) | PC findet die Klasse: **3 Kanten gerichtet, 2 ungerichtet**, 3 Diagramme passen, die möglichen Wirkungen reichen von **0,62 bis 0,99** (wahr 0,60). | `test_default_sample` |
| Wie viele Daten, welches α? (Welt oben) | α = 0,01: CPDAG genau in **50 %** (n = 100), **98 %** (300), **99 %** (1 000), 100 % (3 000), 99 % (10 000). α = 0,05: 74 %, 93 %, 92 %, 93 %, **94 %**; auch bei n = 10 000 im Mittel **0,07 überzählige Kanten je Lauf** (bei α = 0,01 höchstens 0,013): α legt die Fehlalarmrate fest, nicht die Stichprobe. X → Y wird bei n ≥ 300 in 94 bis 100 % der Läufe gerichtet gefunden. | `test_experiment_1_sample_and_alpha` |
| Und mit einer fast untreuen Kante? (Z → X 0,7, X → Y 1,0, Z → Y 0,8, α = 0,05) | Treue-Spielraum **0,048** gegen **0,267** in der Welt oben; PC verwirft Z–X in **84 / 55 / 11 %** der Läufe (n = 300 / 1 000 / 3 000), CPDAG genau nur **14 / 44 / 82 %**. | `test_experiment_1_unfaithful_world` |
| Wie viel bleibt offen? (Zufalls-DAGs, Orakel-CPDAG, 200 je Zelle, 6 / 8 / 10 Knoten) | Anteil **ungerichteter** Kanten bei mittlerem Grad 1: 56 / 56 / 56 %, Grad 2: 42 / 36 / 31 %, Grad 3: 38 / 24 / 23 %. **Vollständig gerichtet** sind nur **4 bis 15 %** der Graphen (Grad 2: 12 / 5 / 6,5 %). Dünne Graphen haben kaum v-Strukturen und bleiben meist offen. | `test_experiment_2_what_stays_open` |
| Und die Wirkung in der Welt? | Die Klasse hat **3 Diagramme**, die Wirkungen sind **0,60 oder 0,98** (0,60 bei Anpassung an Z, 0,60 + 0,60 · 1,312 / 2,0496 = 0,9841 ohne Eltern von X). Aus Daten: Bereich **[0,60; 0,985], Breite 0,38 / 0,38 / 0,39** bei n = 300 / 1 000 / 3 000: **mehr Daten verengen ihn nicht.** | `test_experiment_2_effect_range` |
| Und mit Vorwissen („Saison und Wetter sind Wurzeln“)? | Alle fünf Kanten gerichtet, Klasse mit **1 Diagramm**, Wirkung **0,60** eindeutig, Breite **0,00** in allen Zellen. | `test_experiment_2_effect_range` |
| Wenn der direkte Pfeil den Umweg aufhebt (X → M → Y, direkter Pfeil = −λ · a · b), α = 0,05 | Bei λ = 1 fehlt die Kante X–Y in **96 %** (n = 500) und **97 %** (n = 5 000) der Läufe, und in jedem dieser Läufe setzt PC einen **Collider X → M ← Y, den es nicht gibt** (CPDAG richtig nur 4 und 3 %). Die Zone schrumpft mit n: λ = 0,9 → **68 %** (n = 500) gegen **0 %** (n = 5 000); λ = 0,8 → 13 % gegen 0 %. Bei λ = 0 (kein direkter Pfeil) stimmt der CPDAG in 96 / 92 %. | `test_experiment_3_cancellation` |
| Wenn eine Kante quadratisch wirkt (Z → Y, symmetrisch, lineare Korrelation 0), α = 0,01 | Die Kante Z–Y fehlt in **98 / 98 / 97 / 97 %** der Läufe (n = 300 / 1 000 / 5 000 / 20 000), CPDAG genau höchstens 3 %; linear: 0 % und 98 bis 99 %. Fisher-z sieht nur Linearität, **mehr Daten helfen nicht**. | `test_experiment_3_nonlinear_and_hidden` |
| Wenn die Nachfrage verborgen ist (PC auf S, X, Y, W), α = 0,01 | In **99,3 %** (n = 1 000) und **99,7 %** (n = 5 000) der Läufe dieselbe Ausgabe: **S–X, S → Y, W → Y, X → Y**. X → Y ist richtig gerichtet, S → Y ist falsch; die möglichen Wirkungen reichen von **0,89 bis 0,98**, die Wahrheit **0,60 liegt in 0 von 300** Läufen darin. | `test_experiment_3_nonlinear_and_hidden` |
| Ist der Orakel-PC richtig? | Für alle **543 Diagramme mit vier Knoten**: Orakel-PC = CPDAG der Aufzählung in **543 von 543**; Aufzählung der Klasse = Klasse aus den Aussagenmengen in **543 von 543**; 185 Klassen, davon **59** mit nur einem Diagramm (11 % der 543 Diagramme), größte Klasse 24. | `test_experiment_4_procedure` |
| Zufalls-DAGs mit 6 Knoten (Grad 2), Stichprobe, α = 0,05 | CPDAG genau **23,5 / 47,5 / 67 %** (n = 100 / 1 000 / 10 000), Strukturabstand 3,4 / 2,3 / 1,3. Die Welt oben ist **leicht**; zufällige Welten sind es nicht. | `test_experiment_4_procedure` |
| Hängt das Ergebnis von der Reihenfolge der Variablen ab? (sechs Reihenfolgen je Datensatz) | **Skelett**, ursprüngliche Fassung: 13,5 / 8,5 / 5 %; **stabil: 0 %** (Konstruktion). **Ausgabe** (Orientierung): ursprünglich 54,5 / 34,5 / 19,5 %, stabil **47,5 / 31,5 / 18,5 %**: Stabil behebt das Skelett, nicht die Orientierung. | `test_experiment_4_procedure` |

Handwerte des Textes (`test_hand_values_of_the_text`): Fisher-z bei r = 0,3, n = 103: Prüfgröße **3,095**, p = **0,001967**; kritischer Betrag von r bei α = 0,05: **0,1935**; Teilkorrelation r_xy = 0,5, r_xz = 0,6, r_yz = 0,8: **0,0417**; Kette A → B → C: Klasse **3**, Collider: **1**, Dreieck: **6**; Aufhebung: Kov(X, Y) = **0**.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Treue** | Hebt sich ein Umweg zufällig auf oder ist eine Kante fast untreu, fehlt sie, und PC setzt einen Collider, den es nicht gibt (Experimente 1 und 3). | Sensitivität und Falsifikation |
| **Linear und Gauß (Fisher-z)** | Eine nichtlineare Kante ohne lineare Korrelation bleibt unsichtbar, auch bei vielen Daten (Experiment 3). | nicht gebaut |
| **Alles Wichtige ist gemessen** | Eine verborgene Ursache gibt einen sauber gerichteten Graphen mit einem Wirkungsbereich, der die Wahrheit nicht enthält (Experiment 3). | Wann geht es gar nicht? |
| **Die Klasse genügt** | Ungerichtete Kanten bleiben, die Wirkung ist ein Bereich; mehr Daten ändern das nicht, Vorwissen schon (Experiment 2). | Kontinuierliches Strukturlernen; Über die Äquivalenzklasse hinaus |
| **Die Reihenfolge der Variablen ist gleich** | Das Skelett der stabilen Fassung ist unabhängig, die Orientierung nicht (Experiment 4). | nicht gebaut |
| **Kleine Graphen** | Fünf bis zehn Knoten, ein Test je Aussage ohne Mehrfachtest-Korrektur, Konflikte der Orientierung „wer zuerst kommt“. | nicht gebaut |

Die Befunde, die **Konstruktion** sind, stehen als solche in `test_claims.py`: dass die Klasse das wahre Diagramm enthält (der Wirkungsbereich also die Wahrheit „enthält“; der Befund ist die Breite, die mit n nicht schrumpft), dass „Kante fehlt“ und „falscher Collider“ in jedem Lauf zugleich auftreten (fehlt X–Y mit leerer Trennmenge, ist M nicht in ihr), dass das stabile Skelett nicht von der Reihenfolge abhängt und dass der Orakel-PC den Aufzählungs-CPDAG trifft (Prüfung der Implementierung, kein Befund über die Welt).
Die erste Fassung der Welt hatte Z → X = 0,7, X → Y = 1,0, Z → Y = 0,8: Dort war die Kante Z–X fast untreu (Spielraum 0,048), PC verwarf sie in 55 % der Läufe bei n = 1 000, und „PC braucht n ≈ 3 000“ wäre ein Befund über diese Koeffizienten gewesen, nicht über PC. Die Welt oben (Spielraum 0,267) ist deshalb Voreinstellung, die erste Fassung das Preset „Fast untreu“.
Der Wirkungsbereich folgt der **Idee** von Maathuis/Kalisch/Bühlmann 2009, vereinfacht auf eine Wirkung und kleine Klassen (Aufzählung der Orientierungen, höchstens 12 ungerichtete Kanten). PC, die Meek-Regeln, Verma/Pearl 1990 und PC-stable (Colombo/Maathuis 2014) sind aus dem Gedächtnis geschrieben und nicht gegen die Quellen geprüft; die Implementierung ist durch die exakte Gegenprobe belegt.

## Bezug zu den Nachbarn im Portfolio

- [d-separation-demo](https://github.com/sebastian-hanisch/d-separation-demo): zählt die Äquivalenzklassen (11 für drei, 185 für vier Knoten) und prüft einzelne Aussagen mit Fisher-z; **der PC-Algorithmus gehörte ausdrücklich zu diesem Stück**. `dsep_graph.py` und `equivalence_classes` kommen wortgleich mit und sind das Orakel.
- [backdoor-demo](https://github.com/sebastian-hanisch/backdoor-demo), [adjustierungsmenge-demo](https://github.com/sebastian-hanisch/adjustierungsmenge-demo), [identifizierbarkeit-demo](https://github.com/sebastian-hanisch/identifizierbarkeit-demo): setzen das Diagramm voraus; die Anpassung an die Eltern von X ist die Brücke zum Wirkungsbereich.
- [sensitivitaet-demo](https://github.com/sebastian-hanisch/sensitivitaet-demo): fragt, wie stark eine verborgene Ursache sein müsste; hier der Gegenpart, was PC bei verborgener Ursache ausgibt.
- [uebertragbarkeit-demo](https://github.com/sebastian-hanisch/uebertragbarkeit-demo): selbe Welt, selber Generator; dort ist das Diagramm bekannt.

## Tests

Hand-Instanzen (Fisher-z, Teilkorrelation, Kette, Collider, Dreieck, Meek R1 bis R3, die Welt mit CPDAG und Klasse, Wirkungen 0,6 und 0,9841), Orakel-PC gegen die Aufzählung über **alle 543 Diagramme** mit vier Knoten und Zufalls-DAGs (auch mit Vorwissen), die Teilkorrelation gegen Regressionsresiduen und scipy, der Treue-Spielraum gegen einen zweiten Weg, Skelett-Unabhängigkeit der stabilen Fassung, Auswertung, Verdikte, jede README-Zahl in `test_claims.py`, AppTest (Presets, Regler, Baukasten, Experimente, Permalink, PDF), Footer und Laufzeit-Importe.

```bash
python -m pytest tests -q
```

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `kausal_generator.py` | gemeinsamer Generator der Linie (SplitMix64, Normalverteilung, Strukturmodell) |
| `dsep_graph.py` | d-Separation und Äquivalenzklassen (wortgleich aus Stück 2) |
| `stl_world.py` | Welt mit fünf Größen, Daten, Bevölkerungskovarianz, Treue-Spielraum, Zufalls-DAGs |
| `stl_pc.py` | Fisher-z, PC (Skelett, v-Strukturen, Meek, Vorwissen), Klasse durch Aufzählung, Wirkungsbereich |
| `stl_evaluation.py` | Einstellungen, Analyse, vier Experimente, Baukasten, Verdikte |
| `stl_visualization.py`, `stl_pdf_export.py` | Figuren, Tabellen, PDF |
| `stl_constants.py`, `stl_presets.py` | Regler, Experimente, Presets, Permalink |
| `tests/` | Hand-Instanzen, Orakel, Claims, App |

## Bewusst nicht umgesetzt

- **Latente Variablen** (FCI, MAGs): eine verborgene Ursache wird gezeigt, nicht behandelt.
- **Konservatives PC, GES und andere Score-Verfahren:** nur im Grenzen-Absatz genannt; NOTEARS ist ein eigenes Stück.
- **Nichtlineare und kernbasierte Tests, diskrete Daten, hohe Dimension, Zeitreihen:** der Fisher-z-Test ist der einzige Test.
- **Mehrfachtest-Korrektur und Konfliktbehandlung der Orientierung** wie in der Literatur: α je Test, bei Konflikten gewinnt, wer zuerst kommt.
- **Praxisdaten:** nicht gebaut; alle Zahlen sind erzeugt.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python 3.12.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
