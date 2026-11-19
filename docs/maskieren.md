# Anleitung: Anlagen und Dachelemente sauber maskieren

Stand 19.11.2026. Gilt für den Masken-Editor von PV-Mapper und die Datensätze für Dach-PV.

## Grundsatz: nur Module, keine Zwischenräume

PV-Mapper berechnet die Leistung in zwei Schritten.

1. Die Segmentierung sagt, welche Pixel Module sind.
2. Die Typologie fasst die Pixel zu Anlagen zusammen. Sie misst den Reihenabstand aus den Lücken zwischen den Reihen und setzt daraus den
   Belegungsgrad (`typology.yml`).

Eine Maske, die die Lücken mit einschließt, verdirbt beide Schritte. Das Modell lernt, dass Kies oder Dachpappe zwischen den Reihen „PV“ ist,
und die Reihen lassen sich nicht mehr messen. Die Leistung wird dann bei aufgeständerten Anlagen um bis zu das Doppelte überschätzt.

## Regeln für PV-Module

| Situation | So maskieren |
|---|---|
| Schrägdach, Module dicht an dicht | Ein Polygon um das ganze Modulfeld, Außenkante der Modulrahmen |
| Fugen zwischen Modulen einer Reihe (< 20 cm) | mitnehmen, also die Reihe als ein Rechteck zeichnen |
| Aufgeständerte Reihen auf dem Flachdach | **jede Reihe einzeln**, Lücken zwischen den Reihen frei lassen |
| Ost-West-Doppelreihen (Satteldach-Form) | jedes Doppel als ein Rechteck; Wartungsgänge dazwischen frei |
| Schornstein, Lüfter, Dachfenster **im** Modulfeld, größer als etwa 0,5 m² (≈ 70 × 70 cm) | aussparen, also das Polygon darum herumführen oder das Element als eigenes Polygon in der passenden Klasse zeichnen |
| kleinere Durchdringungen | ignorieren, also im Modulfeld lassen |
| Schatten der Reihen (nördlich jeder Reihe) | **nicht** mitnehmen; er ist kein Modul |
| Module teilweise von Bäumen verdeckt | nur den sichtbaren Teil |
| Gebäude „kippt“ im Luftbild (Hochhaus, kein True-Ortho) | Module dort zeichnen, wo sie im Bild zu sehen sind, nicht dort, wo sie auf dem Grundriss liegen müssten |
| Anlage reicht über den Gebäudeumriss hinaus (Carport, Vordach) | mitzeichnen; der Rand gehört zum Bild |

Werkzeuge:

- **Rechteck (Kante–Kante–Tiefe):** für Reihen. Es ist schneller und genauer als Freihand.
- **Fang:** hilft, Ecken benachbarter Reihen sauber zu treffen.

## Andere Dachelemente

Andere Dachelemente bekommen eine eigene Klasse. Ab der neuen Fassung des Editors wählt man sie im Klassenwähler statt mit dem Umschalter
„PV / Gaube“. Gezeichnet wird nur, was **auf diesem Dach** mit PV verwechselt werden könnte, nicht jedes Element.

| Klasse | Typische Merkmale | Zeichnen |
|---|---|---|
| Solarthermie | wenige, kleinere Kollektoren (2–6), oft bläulich-dunkel, Röhrenstruktur, Leitungen zum Dach | ganze Kollektorfläche |
| Gaube | eigener kleiner Dachaufbau mit Fenster, oft dunkles Dach oder Schiefer | Dachfläche und Front der Gaube |
| Dachfenster / Lichtkuppel | rechteckig, glänzend oder spiegelnd, oft in Reihen; Lichtkuppeln rund oder gewölbt | jedes Element |
| Technik | Lüftungsgeräte, Klimaanlagen, Antennen, Aufzugsaufbauten | Umriss |
| dunkle Dachfläche | Bitumen-Flicken, nasse Stellen, dunkle Ziegel, die wie Module aussehen | nur den verwechselbaren Teil |
| Schatten | Schlagschatten von Aufbauten oder Nachbargebäuden, wenn er wie eine Modulfläche aussieht | Schattenfläche |
| Begrünung | Gründach, Moos | nur wenn verwechselbar |

## Urteil je Bild

- **PV vorhanden:** Module sind im **aktuellen** Bild sicher zu erkennen. Alle Modulfelder zeichnen.
- **Kein PV:** sicher keine Module. Verwechselbare Elemente in ihrer Klasse zeichnen; daraus lernt das Modell am meisten.
- **Unklar:** Das Bild ist zu unscharf, verdeckt oder widersprüchlich → „überspringen“. Bitte nicht raten. Ein falsches Label schadet mehr
  als ein fehlendes.
- **Teilweise:** Neben echten Modulen liegt etwas Verwechselbares, zum Beispiel PV und Solarthermie auf einem Dach. Beides in seiner Klasse
  zeichnen; das Bild bleibt „PV vorhanden“.

## Schneller Ablauf je Bild

1. Ein Blick aufs ganze Bild: Gibt es Module? Wenn nein, verwechselbare Elemente zeichnen und als „Kein PV“ speichern.
2. Modulfelder mit dem Rechteck zeichnen, Reihe für Reihe.
3. Größere Aussparungen und andere Klassen ergänzen.
4. Speichern. Gespeichert gilt als geprüft.
