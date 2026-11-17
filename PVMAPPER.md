# PV-Mapper: DeepPVMapper für Europa

Diese Erweiterung von DeepPVMapper erkennt Solaranlagen auf Dächern in **Deutschland, den Niederlanden, Österreich und Frankreich**. Die Luftbilder kommen direkt aus den offenen Orthofoto-Diensten (WMS/WMTS) der Länder. Bedient wird das Ganze über eine **lokale Web-Oberfläche**. Die ursprüngliche Pipeline für Frankreich (`main.py`) funktioniert weiterhin unverändert.

| | Original (`main.py`) | PV-Mapper (`python -m pvmapper`) |
|---|---|---|
| Länder | Frankreich (IGN-Kacheln) | DE (alle 16 Länder), NL, AT, FR, weitere per YAML |
| Luftbilder | JP2-Download pro Département | WMS/WMTS-Abruf nur für benötigte Ausschnitte |
| Modus | flächendeckend | **Gebäudeliste** oder **Fläche** (Rechteck/Polygon) |
| Größen | nur 1,7–36,1 kWp | alle; Größenklassen und kWp-Filter frei einstellbar |
| Bedienung | Kommandozeile | Web-Oberfläche und Kommandozeile |
| Abbruch | Fortsetzen pro Département | Fortsetzen pro Gebäude bzw. pro Kachel |
| GPU | CUDA | CUDA, **AMD ROCm**, Apple MPS, CPU |

## Installation (Ubuntu, AMD Radeon)

```bash
git clone -b pvmapper-eu https://github.com/chri6o/deeppvmapper
cd deeppvmapper
./install_pvmapper.sh rocm        # oder: cuda | cpu
source .venv/bin/activate
python -m pvmapper web            # → http://127.0.0.1:8765
```

Das Skript installiert PyTorch für ROCm (die ROCm-Laufzeitbibliotheken sind in den PyTorch-Paketen enthalten) und die übrigen Pakete. Danach lädt es die Modellgewichte (~630 MB) von Hugging Face. Ob die GPU erkannt wurde, zeigt `python -m pvmapper device` und oben rechts die Oberfläche. Wird die RX 7600 XT nicht erkannt, hilft vor dem Start meist `export HSA_OVERRIDE_GFX_VERSION=11.0.0`. GDAL braucht PV-Mapper nicht.

## Eigener PC als PV-Mapper-Server (SSD am Mac vorbereiten)

`deploy/ssd/prepare_ssd_mac.sh` richtet eine externe SSD am Mac als fertiges Boot-Laufwerk ein. Empfohlen ist `./prepare_ssd_mac.sh --usb-stick`: Das Skript schreibt einen Ubuntu-Installationsstick und legt die Konfiguration auf die SSD. Am PC vom Stick starten, einmal `yes` tippen, und der PC installiert in 10–20 Minuten selbst auf die SSD. Das braucht Stick, Monitor und Tastatur. Ohne Option installiert das Skript in einer virtuellen Maschine auf dem Mac; auf Apple Silicon ist das sehr langsam.

1. **Am Mac:** SSD anschließen, Terminal öffnen.
   ```bash
   git clone -b pvmapper-eu https://github.com/chri6o/deeppvmapper && cd deeppvmapper/deploy/ssd
   cp pvmapper-ssd.conf.example pvmapper-ssd.conf   # optional: Zugangsdaten vorab eintragen
   ./prepare_ssd_mac.sh
   ```
   - Das Skript braucht Homebrew und installiert QEMU, falls es fehlt.
   - Es lädt Ubuntu Server 24.04 LTS und prüft die Prüfsumme.
   - Es fragt nach Rechnername, Linux-Passwort, Browser-Passwort, PV-Map-Token, Claude-/OpenAI-Schlüssel und optional WLAN. Fehlende Werte lassen sich später in den Einstellungen nachtragen.
   - Danach installiert es Ubuntu unbeaufsichtigt in einer virtuellen x86-Maschine direkt auf die SSD. Auf Apple Silicon dauert das 30–90 Minuten, auf einem Intel-Mac etwa 10 Minuten.
   - Auf der SSD landen: Zugangsdaten, `pvmapper.yml`, Netzwerk (Kabel per DHCP, optional WLAN), SSH mit dem Schlüssel des Macs und Avahi (`pvmapper.local`).
2. **Am PC:** SSD einbauen oder als USB-SSD an einen USB-3-Port hinten am Mainboard stecken (nicht an Front- oder Hub-Anschlüsse), Netzwerkkabel stecken, einschalten. Einmalig von USB starten geht über das Boot-Menü (je nach Hersteller F8, F11 oder F12). Den USB-Stromsparmodus schaltet das System ab (`usbcore.autosuspend=-1`), damit die Platte im Betrieb nicht wegbricht. Falls nötig im UEFI (Entf/F2) die SSD als erstes Boot-Laufwerk wählen; der UEFI-Modus muss aktiv sein, Secure Boot darf an bleiben.
3. **Im Browser (Mac oder anderer Rechner im Netz):** `http://pvmapper.local:8765`. Beim ersten Start zeigt die Seite die Einrichtung (PyTorch für ROCm, PV-Mapper, Modelle; 10–30 Minuten), danach PV-Mapper. Die Einrichtung prüft die Radeon und setzt bei Bedarf `HSA_OVERRIDE_GFX_VERSION=11.0.0`.

Danach: `ssh pvmapper@pvmapper.local`. Aktualisieren mit `sudo pvmapper-update`, Protokolle mit `journalctl -u pvmapper -f`, die Einrichtung steht in `/var/log/pvmapper-firstboot.log`. Mit gesetztem Browser-Passwort fragt die Oberfläche danach; der Benutzername ist egal.

## Oberfläche

- **Gebäude prüfen:** Gebäudedatei hineinziehen (GeoJSON, CSV mit `lat`/`lon` oder WKT, GeoPackage, Shapefile) oder eine PostGIS-Abfrage eingeben. Dann ID-Feld, Bildquelle und Modell wählen und starten. Für jedes Gebäude ergibt sich: PV ja/nein, PV-Fläche, geschätzte kWp, Größenklasse, Score und ein Vorschaubild.
- **Fläche auswerten:** Rechteck oder Polygon in die Karte zeichnen. Das Gebiet wird in Kacheln zerlegt und vollständig ausgewertet. Optional kann ein Gebäudelayer angegeben werden, um die Treffer Gebäuden zuzuordnen.
- **Aufträge:** Fortschritt, Durchsatz, Restzeit und Protokoll. Abbrechen und Fortsetzen ist jederzeit möglich. Ergebnistabelle mit Filtern; ein Klick zeigt Gebäude und Vorschaubild (links das Original, rechts die erkannte PV in Rot, gelb der Gebäudeumriss). Exporte: `buildings.csv`, `buildings.geojson`, `pv_polygons.geojson` bzw. `installations.*` und `summary.json`.
- **Probe:** In die Karte klicken. Das Luftbild wird geladen und sofort analysiert. Das eignet sich, um Quellen und Modelle pro Land zu vergleichen, bevor ein großer Auftrag startet.
- **Bildquellen:** alle hinterlegten Dienste mit Auflösung und Lizenz, auf Wunsch als Ebene in der Karte.
- **Einstellungen:** Modell, Gerät, Schwellenwerte, Batchgröße, parallele Downloads, kWp pro m², Mindestfläche, kWp-Filter und **Größenklassen** (Standard: S < 30, M < 100, L < 750, XL ≥ 750 kWp).

## PV-Map-Konnektor und Training mit Bestands-PV

PV-Mapper holt aus PV-Map **Gebäudelisten** und **Trainingsbeispiele**. In PV-Map legst du dafür ein API-Token an (Verwaltung → Datenquellen → API-Tokens) und trägst es mit der Adresse in den Einstellungen von PV-Mapper ein. Schnittstelle: PV-Map `docs/pvmapper-connector.md`.

- **Gebäude prüfen → „aus PV-Map laden“:** nach Land, Region (NUTS3/LAU), Gebäudeart und PV-Stand („ohne bekannte PV“ für die Suche, „mit bekannter PV“ für den Abgleich). Gebäudeart, Name und der PV-Stand aus PV-Map kommen in die Ergebnisse mit. Die Datei `comparison.json` zeigt, wie viele der laut Register belegten Gebäude die Erkennung findet. Auf der Kommandozeile ist die Eingabe z. B. `pvmap:cc=DE&lau=DE_11000000&sub=Education&pv=none`.
- **Training:** PV-Map liefert eine Stichprobe, geschichtet nach **Gebäudeart × Anlagengröße** (kWp-Grenzen frei wählbar, Standard 10/30/100/750). Die Labels sind „PV sicher“ (Register A/B, Meldung), „PV wahrscheinlich“ (Register C/D, Erkennungsebene), „keine PV (gemeldet)“ und „keine PV bekannt“ (schwach).
  1. Ein Viertel der Gebäude bleibt zur Bewertung zurück.
  2. Schwache Negative, bei denen das Ausgangsmodell sehr sicher PV sieht, werden verworfen, denn dort gibt es wahrscheinlich unregistrierte PV.
  3. Der Klassifikator wird per *Multiple-Instance-Learning* trainiert: Das Label gilt für das Gebäude, also für den besten Bildausschnitt. Damit funktionieren kleine und große Dächer gleich.
  4. Die Segmentierung wird nur nachtrainiert, wenn genug PV-Polygone vorliegen (Erkennungsebenen in PV-Map).
  5. Die Schwelle wird auf F1 optimiert, kWp/m² je Größenklasse kalibriert.
  6. Gespeichert wird das neue Modell nur, wenn es auf der Validierung mindestens so gut ist wie das Ausgangsmodell, und zwar als Profil `custom:<name>`. Der Bericht vergleicht vorher und nachher nach Art, Größe und Label.
- Statt PV-Map geht auch eine eigene GeoJSON mit `properties.label` (0/1).

## Masken zeichnen und KI-Vorschläge

**Datensatz anlegen:** Im Reiter „Training“ den Knopf „Nur Datensatz anlegen“ drücken. Die Software lädt Stichprobe und Luftbilder und trainiert noch nicht.

**Reiter „Masken“:** links der Datensatz mit Kennzahlen und Filtern (PV ohne Maske, KI ungeprüft, offen, fertig, übersprungen, Art, Größe), rechts der Editor. Klick auf eine Zeile öffnet das Bild, Gebäudeumriss gelb gestrichelt.

| Taste | Funktion |
|---|---|
| `1` / `2` / `3` | Polygon / Rechteck / gedrehtes Rechteck (zwei Klicks für eine Kante, dritter für die Breite) |
| Doppelklick, `Enter` | Polygon schließen; `Enter` ohne offenes Polygon: speichern und weiter |
| `M` / `K` | Vorschlag des Modells / der KI übernehmen (danach korrigieren) |
| `Entf`, `Strg+Z` | ausgewählte Fläche löschen, rückgängig |
| `L`, `X` | Label „PV vorhanden“ umschalten (korrigiert das Register-Label), Bild überspringen |
| `←` / `→`, Mausrad, Leertaste+Ziehen | blättern, zoomen, verschieben; `F` = einpassen |

Masken landen als `dataset/<id>.mask.png`, die Herkunft steht in der zugehörigen JSON-Datei (von Hand, KI, PV-Map-Ebene). Danach „Training mit diesem Datensatz“ starten. Schon ab 20 Masken wird auch die Segmentierung nachtrainiert.

**KI (Claude/ChatGPT):** Einen API-Schlüssel und das Modell in den Einstellungen eintragen. Es gibt zwei Modi:
- **KI prüft Modellvorschläge (Standard):** Das eigene Modell schlägt Flächen mit niedriger Schwelle vor. Sie werden nummeriert eingezeichnet, und die KI sagt, welche Nummern PV sind; fehlende Flächen darf sie ergänzen. Die Umrisse bleiben pixelgenau, die Entscheidung trifft ein unabhängiges Modell.
- **KI zeichnet selbst:** Die KI liefert die Polygone direkt. Das ist gröber und nur für Fälle gedacht, in denen das Modell nichts findet.

Einzeln geht das per Taste `K`, für viele Bilder über „KI-Masken für viele Bilder“ (als eigener Auftrag, mit Tokenzählung). KI-Masken sind **ungeprüft**, bis du sie im Editor speicherst. Das Training nutzt standardmäßig nur geprüfte, die Bewertung immer nur Register- und Meldungs-Labels. API-Schlüssel stehen im Klartext in `pvmapper.yml`; die Oberfläche zeigt sie nicht an.

## Erkennungsqualität: Nachbearbeitung, Gegenprobe, Verwechsler

Die Modelle markieren ohne Nachtraining oft Gauben, Dachfenster oder helle Dachelemente als PV und erfassen
Modulreihen mit Reflexen oder Schatten nur teilweise. Drei Bausteine verbessern das (Einstellungen, alle an):

- **Nachbearbeitung** (`pvmapper/postproc.py`): nur Flächen innerhalb des Gebäudeumrisses (+1,5 m) zählen.
  Verworfen werden Flächen ohne Kontrast zum Dach (nicht dunkler als die Umgebung; nur kompakte, gleichmäßige,
  deutlich blauere Flächen dürfen so hell sein), unregelmäßige Flecken (z. B. Schieferdächer, Dachflächen im Licht) und Reihen
  kleiner, gleichartiger Flecken ohne größeres Feld (Gauben, Lichtkuppeln). Erkannte Felder werden entlang
  gleichfarbiger Pixel ergänzt, gleichfarbige rechteckige Felder auf demselben Dach (ab 4 m²) hinzugenommen,
  Löcher durch Reflexe gefüllt und fast rechteckige Felder begradigt. Die Ergänzung greift nur, wenn sich die
  Module farblich klar vom Dach abheben. Im Steckbrief steht unter „Verworfen“, was entfernt wurde.
  Schwellen: `postproc_*` in `pvmapper.yml`.
- **Gegenprobe mit zweitem Luftbild**: wo die Quelle ältere Jahrgänge hat (NRW 2022–2024, NL 2023–2025),
  wird dasselbe Fenster aus einem anderen Jahrgang geladen und ebenfalls ausgewertet. Felder, die dort nicht
  auftauchen, fallen weg (Schatten, Reflexe); Felder, die dort vollständiger zu sehen sind, werden ergänzt.
  Zeigt das zweite Bild gar keine PV, bleibt das Ergebnis (evtl. neue Anlage) – Status „im Vergleichsbild keine
  PV“. Die Vorschau zeigt rechts das Vergleichsbild. Kostet etwa die doppelte Ladezeit.
- **Segmentierung mit gespiegelten Bildern** und Segmentierung auch der schwächeren Nachbar-Ausschnitte in
  Bildern mit PV (`neighbor_factor`): vollständigere Reihen.

Für das Training gibt es im Masken-Editor die Ebene **„Kein PV“** (Taste N): Gauben, Dachfenster, Solarthermie
markieren. Das Training nutzt diese Stellen als harte Gegenbeispiele (Klassifikator) und gewichtet sie in der
Segmentierung dreifach; Bilder nur mit „Kein PV“-Markierung werden ebenfalls verwendet. Zusätzlich variiert das
Training Helligkeit, Gamma, Schattenbänder und Reflexe. Die KI-Masken prüfen jede Fläche gegen eine Liste
typischer Verwechsler und durchlaufen dieselbe Nachbearbeitung.

### Masken-Editor: Werkzeuge und Tasten

Der Editor nutzt das ganze Fenster (☰ oder B blendet die Liste ein). Werkzeuge: V Zeiger (Fläche wählen und verschieben,
Ecken ziehen, Doppelklick auf eine Kante fügt eine Ecke ein, Entf löscht Ecke bzw. Fläche), H Hand, Z Lupe (Klick vergrößert,
Rechtsklick verkleinert, Rahmen ziehen), 1 Polygon, 2 Rechteck, 3 gedrehtes Rechteck. Halten: Shift oder Leertaste = Hand,
Cmd/Strg = Zeiger, Alt = Lupe. Ebenen: N oder G wechselt zwischen PV und „Gaube / kein PV“. Fang (S): Ecken aus dem Luftbild,
Ecken und Kanten der Flächen und des Gebäudeumrisses. ? zeigt alle Tasten.

### Erkennungen als Trainingsbeispiele

In den Ergebnissen von „Gebäude prüfen“ lassen sich Gebäude anhaken (oder „Zufällig wählen“, z. B. 40) und mit
**„Als Trainingsdatensatz →“** übernehmen. Die Luftbilder werden geladen, die erkannten PV-Flächen sind als
Vorschlag eingezeichnet (Quelle „Erkennung ?“). Im Reiter **Masken** (Filter „Erkennung ungeprüft“) jede Maske
korrigieren und mit Enter speichern; Fehltreffer mit „PV vorhanden“ (L) zum Gegenbeispiel machen und Gauben o. Ä.
auf der Ebene „Kein PV“ markieren. Ins Training gehen nur gespeicherte (geprüfte) Bilder; Gegenbeispiele lädt das
Training bei Bedarf aus PV-Map nach.

## Anlagentypologie und Leistung je Anlage

Für jede erkannte Anlage bestimmt `pvmapper/typology.py` aus Maske und Luftbild (ohne weiteres neuronales Netz):

- **Anlage und Umriss:** Modulflächen mit weniger als 3 m Abstand bilden eine Anlage; der Umriss schließt die Lücken zwischen den Reihen.
- **Bauform:** `flush` (flächig, auch Modulraster ohne Reihen), `tilted_mono` (aufgeständert, Reihen mit Abstand), `east_west` (dichte Paare mit zwei Helligkeitsstufen, Flächen nach Ost und West), `vertical_bifacial`, sonst `unknown`.
- **Ausrichtung** aus der Reihennormalen (Süd-Reihen: Seite ohne Schattenstreifen), **Neigungsklasse** nach MaStR (`<5`, `5-20`, `21-40` …) als Typ-Standard mit niedriger Konfidenz.
- **Belegungsgrad:** gemessen aus Reihenabstand und Deckung (`derived_pitch`), sonst Standard der Bauform (`type_default`, siehe `pvmapper/typology.yml`).
- **Leistung:** Umriss × Belegungsgrad / cos(Neigung) × Modulwirkungsgrad (nach Luftbildjahr). Dazu eine Spanne `kwp_low`–`kwp_high`.

Was von oben nicht sichtbar ist (gebäudeintegriert, Befestigung, Unterkonstruktion, Modultechnik), bleibt `unknown`. Jedes Merkmal trägt eine Konfidenz.

Die Typologie läuft überall, wo Anlagen gefunden werden:

- **Gebäude prüfen** und **Aufträge** (Nachtabgleich): je Gebäude.
- **Fläche auswerten:** je Zelle, in der Anlagenliste mit Bauform und Spanne; Umrisse in `installation_outlines.geojson`.
- **Probe:** Bauform und kWp je Anlage unter dem Bild, Umrisse auf der Karte.
- **Masken:** Der Editor zeigt Bauform und kWp der gerade gezeichneten Maske und ihre Umrisse gestrichelt. Von Hand geprüfte Masken gehen mit dieser Schätzung an PV-Map (mit `typology_push` auch mit den Anlagen).

Ausgabe beim Gebäude-Prüfen: Spalten `kwp_low`, `kwp_high`, `layout`, `kwp_method` (`typology` oder `area_factor`) in `buildings.csv`, dazu `installations.geojson` mit einem Umriss je Anlage, allen Merkmalen sowie OSM-Tags (`osm`) und MaStR-Feldern (`mastr`). In der Ergebnisliste zeigt die Vorschau je Anlage Bauform, Ausrichtung, Belegung und kWp.

Abschalten: Einstellungen → „Anlagentyp und Leistung je Anlage“ (dann wie bisher Modulfläche × `kwp_per_m2`). An PV-Map gehen die Anlagen mit `typology_push: true` in `pvmapper.yml` (PV-Map ab Migration 137 speichert sie und zeigt sie dem Owner im Steckbrief).

**Kalibrierung** an den Registerleistungen (PV-Map liefert `pv_kwp` je Gebäude):

```bash
python -m pvmapper calibrate 20261117-210000-abcd
```

Das schreibt `calibration.json` in den Job: je Bauform der Belegungsgrad, der die Registerleistung ergäbe (Median, Quartile, aktueller Standard), und der Fehler der Typologie-Schätzung im Vergleich zur alten Flächen-Schätzung. Passende Werte trägt man in eine eigene `typology.yml` ein (`typology_file` in den Einstellungen).

## Anlagenalter aus älteren Luftbildern

Für jede gefundene Anlage sucht `pvmapper/age.py` in älteren Befliegungen, ab wann die Module zu sehen sind. Die Halbierung der Zeitspanne braucht dafür 3–5 zusätzliche Bilder je Anlage, nur für Treffer und parallel für viele Gebäude. Ergebnis: „Baujahr ≈ 2018 (2016–2019)“. Das Baujahr bestimmt den Modulwirkungsgrad (`efficiency_by_year` in `typology.yml`) und damit kWp und Spanne.

- **NRW:** historische DOP von Geobasis NRW; welche Flüge es am Ort gibt (Datum, Auflösung), liefert die Metadaten-Ebene. Das Flugdatum des aktuellen Bilds kommt aus der DOP-Metadaten-Ebene.
- **Niederlande:** PDOK-Jahrgänge 2016–2020 (25 cm) und ab 2021 (8 cm).
- Flüge gröber als `age_max_res_m` (0,4 m) bleiben unberücksichtigt.
- **Befund je Flug:** Anteil der heutigen Module, die das Modell im alten Bild findet; Kontrast Module ↔ Dach im Vergleich zu heute; Ähnlichkeit des Helligkeitsmusters. Unklare Flüge werden übersprungen. Jeder Befund steht in `age_flights` der Anlage.
- **Grenzen:** Ist die Anlage schon auf dem ältesten brauchbaren Flug zu sehen, steht da „vor <Jahr>“. Erweiterungen zählen ab dem ersten sichtbaren Teil, wenn sie in derselben Anlage liegen.

Das Register (MaStR-Inbetriebnahmedatum) wird dafür bewusst nicht genutzt; es dient später als Gegencheck.

Abschalten: Einstellungen → „Baujahr aus älteren Luftbildern“. Sichtbar in der Vorschau, im Karten-Popup, in der Tabelle bei „Fläche auswerten“ und bei der Probe.

## Abgleich mit PV-Map (Nachtbetrieb)

Unter **Einstellungen → PV-Map-Abgleich** Nachtbetrieb an, Zeitfenster und Gebäudefilter (z. B. `cc=DE&nuts3=DEA2`) setzen. Im Zeitfenster
holt PV-Mapper die nächsten Gebäude aus der Arbeitsliste von PV-Map (zuerst nie geprüfte), prüft sie und sendet jedes Ergebnis
zurück, auch „keine PV“. Nach der Wiederholfrist wird das Luftbild erneut geladen; nur wenn es sich geändert hat (neue Befliegung,
Fingerabdruck des Bildes), wird neu gerechnet. „Bestätigt“ (zählt in PV-Map als PV, je nach Sichtbarkeit) ist eine Erkennung, wenn die
Gegenprobe mit dem zweiten Luftbild sie bestätigt und der Score über der Schwelle liegt, oder wenn sie im Masken-Editor geprüft wurde
(gespeicherte Masken gehen automatisch an PV-Map). „Jetzt abgleichen“ startet einen Lauf von zwei Stunden. Details und die Sichtbarkeit in
PV-Map: PV-Map `docs/pvmapper-ergebnisse.md`.

### Iteratives Training mit unvollständigen Ergebnissen

1. In den Ergebnissen einer Prüfung den Filter **„Nacharbeit: verpasst / unvollständig“** wählen. Gelistet werden Gebäude mit
   PV im Register, aber ohne Fund („verpasst“), mit deutlich weniger erkannter Leistung als registriert („unvollständig“) und
   ohne Fund trotz hohem Score („Verdacht“).
2. Gebäude anhaken (oder „Zufällig wählen“) und **„Als Trainingsdatensatz →“**: neuer Datensatz oder an einen bestehenden anhängen,
   so wachsen die geprüften Beispiele von Runde zu Runde.
3. Im Reiter **Masken** die vorgeschlagenen Flächen ergänzen und speichern.
4. „Training mit diesem Datensatz“: Ausgangsmodell ist vorausgewählt das Modell, das die Ergebnisse geliefert hat; der Name
   schlägt die nächste Version vor (`…-v2`, `…-v3`). Danach mit dem neuen Modell erneut prüfen und wieder bei 1 beginnen.

Mit Masken lernt der Klassifikator je Bildausschnitt (PV ja/nein) statt nur je Gebäude – das ist das stärkere Signal.

## Kommandozeile

```bash
python -m pvmapper buildings schulen.gpkg --id-field osm_id --sources DE
python -m pvmapper buildings "postgresql://user:pw@localhost/osm" \
       --query "SELECT osm_id AS id, ST_AsText(ST_Transform(way,4326)) AS wkt FROM planet_osm_polygon WHERE amenity='school'"
python -m pvmapper area --bbox 13.30 52.45 13.36 52.48 --kwp-min 100
python -m pvmapper resume 20261115-013936-688d
python -m pvmapper sources --test
python -m pvmapper pvmap --cc DE --strata                      # verfügbare Trainingsbeispiele
python -m pvmapper buildings "pvmap:cc=DE&nuts3=DEA22&pv=none"
python -m pvmapper train --cc DE --sub Education --per-stratum 150 --negatives 150 --name de-schulen
python -m pvmapper calibrate 20261115-013936-688d                # Belegungsgrade gegen Register-kWp
```

Die Einstellungen stehen in `pvmapper.yml` im Arbeitsverzeichnis. Die Oberfläche schreibt diese Datei, Optionen auf der Kommandozeile überschreiben sie.

## Bildquellen

Alle Quellen sind in `pvmapper/sources.yml` eingetragen. Ihre Endpunkte wurden im November 2026 live per GetCapabilities und GetMap geprüft:

- **Deutschland:** alle 16 Länder (DOP10/DOP20; Bayern als Open Data nur DOP40)
- **Niederlande:** PDOK Luchtfoto actueel HR (8 cm) und 25 cm
- **Österreich:** basemap.at Orthofoto
- **Frankreich:** IGN Géoplateforme

Mit **„automatisch nach Lage“** nimmt die Software die passende Quelle anhand der Koordinaten. Liefert eine Quelle ein leeres Bild (außerhalb ihres Gebiets), versucht sie die nächste. Eigene Dienste trägst du in eine zweite YAML-Datei mit gleichem Aufbau ein und gibst deren Pfad in den Einstellungen unter „Zusätzliche Quellen“ an. Vor einer Veröffentlichung der Ergebnisse die Lizenz- und Namensnennungsangaben der Dienste prüfen.

## Modelle

| Profil | Training | Auflösung | Lizenz | Einsatz |
|---|---|---|---|---|
| `nrw` (Standard) | 3D-PV-Locator (Mayer et al. 2022), DOP10 NRW | 10 cm | MIT | DE / NL / AT |
| `ign` | BDAPPV (Kasmi et al.), IGN Frankreich | 20 cm | CC-BY 4.0 | FR |
| `google` | BDAPPV, Google-Luftbilder | 20 cm | Bilder CC-BY-NC | nur nicht-kommerziell |

**Was die Modelle können, ehrlich eingeschätzt:** Die Profile wurden mit Bildern aus *einer* Region trainiert. Große, gut sichtbare Anlagen, etwa auf Flachdächern von Schulen, Hallen oder Verwaltungsgebäuden, erkennt das NRW-Modell im Test zuverlässig, auch die Fläche stimmt gut. Kleine, vollschwarze Module auf dunklen Dächern (typisch für niederländische Reihenhäuser) übersieht es dagegen häufig. Vor einem großen Lauf deshalb pro Land mit „Probe“ und einer Stichprobe von ~200 Gebäuden prüfen. Die Vorschaubilder sind genau dafür da. Dauerhaft besser wird es mit einem Nachtraining auf eigenen Beispielen pro Land (Trainingsskripte im HF-Repository `gabrielkasmi/bdappv-models`).

**kWp-Schätzung:** erkannte Modulfläche (Draufsicht) × `kwp_per_m2` (Standard 0,20). Das ist eine Größenordnung, kein Anlagenregister. Bei Schrägdächern ist die echte Modulfläche größer als die Draufsicht.

## Durchsatz

Pro Gebäude fällt ein WMS-Abruf (~0,3–2 s) plus die GPU-Analyse an. Bei 8 parallelen Downloads ist meist der Abruf der Engpass, nicht die GPU. Grob gerechnet sind das 20.000–30.000 Gebäude pro Stunde, für 1 Mio. Gebäude also 1,5–2 Tage. Die Grenze setzen die öffentlichen Server: `concurrency` und `min_interval_s` pro Quelle in `sources.yml` regeln, wie stark sie belastet werden.

## Tests

```bash
pip install pytest && python -m pytest tests/test_pvmapper.py -q
```

Die Tests kommen ohne Modellgewichte und ohne Netz aus. Sie decken Kachelraster, Koordinatensysteme, Leseroutinen, Größenklassen, die Quellenzuordnung und einen kompletten Gebäude-Job mit Fortsetzen ab.
