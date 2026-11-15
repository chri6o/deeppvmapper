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

## Oberfläche

- **Gebäude prüfen:** Gebäudedatei hineinziehen (GeoJSON, CSV mit `lat`/`lon` oder WKT, GeoPackage, Shapefile) oder eine PostGIS-Abfrage eingeben. Dann ID-Feld, Bildquelle und Modell wählen und starten. Für jedes Gebäude ergibt sich: PV ja/nein, PV-Fläche, geschätzte kWp, Größenklasse, Score und ein Vorschaubild.
- **Fläche auswerten:** Rechteck oder Polygon in die Karte zeichnen. Das Gebiet wird in Kacheln zerlegt und vollständig ausgewertet. Optional kann ein Gebäudelayer angegeben werden, um die Treffer Gebäuden zuzuordnen.
- **Aufträge:** Fortschritt, Durchsatz, Restzeit und Protokoll. Abbrechen und Fortsetzen ist jederzeit möglich. Ergebnistabelle mit Filtern; ein Klick zeigt Gebäude und Vorschaubild (links das Original, rechts die erkannte PV in Rot, gelb der Gebäudeumriss). Exporte: `buildings.csv`, `buildings.geojson`, `pv_polygons.geojson` bzw. `installations.*` und `summary.json`.
- **Probe:** In die Karte klicken. Das Luftbild wird geladen und sofort analysiert. Das eignet sich, um Quellen und Modelle pro Land zu vergleichen, bevor ein großer Auftrag startet.
- **Bildquellen:** alle hinterlegten Dienste mit Auflösung und Lizenz, auf Wunsch als Ebene in der Karte.
- **Einstellungen:** Modell, Gerät, Schwellenwerte, Batchgröße, parallele Downloads, kWp pro m², Mindestfläche, kWp-Filter und **Größenklassen** (Standard: S < 30, M < 100, L < 750, XL ≥ 750 kWp).

## Kommandozeile

```bash
python -m pvmapper buildings schulen.gpkg --id-field osm_id --sources DE
python -m pvmapper buildings "postgresql://user:pw@localhost/osm" \
       --query "SELECT osm_id AS id, ST_AsText(ST_Transform(way,4326)) AS wkt FROM planet_osm_polygon WHERE amenity='school'"
python -m pvmapper area --bbox 13.30 52.45 13.36 52.48 --kwp-min 100
python -m pvmapper resume 20261115-013936-688d
python -m pvmapper sources --test
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
