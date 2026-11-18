# PV-Freiflächen (PV-Mapper-Modul, #377)

Stand: 18.11.2026, Plan freigegeben von Christoph (PV-Map #377). Umsetzung hier im Paket `pvmapper/ground/`; PV-Map übernimmt die
Ergebnisse. Dieselbe Datei liegt in PV-Map unter `docs/freiflaechen.md`, die Plugin-Art `imagery` in PV-Map `docs/plugins.md`.

## Abgrenzung

- Nur PV auf Freiflächen: Solarparks, Agri-PV, Floating-PV. Kandidaten mit mehr als 20 % Überlappung mit OSM-Gebäudegrundrissen werden
  verworfen und nicht weiterverarbeitet (`ground_max_building_share`, Standard 0,2).
- Mindestgröße 0,5 ha (`ground_min_ha`).
- Keine Käufe von Satellitenbildern; optional Export „Ausschnitt anfragen“ (GeoJSON) von Hand.
- Kein Download ganzer Orthofoto-Bestände: nur Ausschnitte um Kandidaten (500 m Puffer).
- Gebäude-Pipeline, Gebäudelisten und Dachanlagen bleiben unverändert.

## Ablauf

1. **Ausgangsbestand** (`ground/inventory.py`): OSM `power=plant` mit `plant:source=solar` und `power=generator` mit
   `generator:source=solar`, ohne `location=roof`/`generator:place=roof` und ohne Flächen, die überwiegend auf Gebäuden liegen; TZ-SAM
   (Zenodo, vierteljährlich). Dubletten: Flächenüberlappung > 50 % der kleineren Fläche oder Abstand < 100 m bei Punkten. Der Bestand
   ist Trainings- und Testmenge.
2. **Grobsuche** (`ground/coarse.py`, monatlich, Auftrag `ground_scan`): openEO im Copernicus Data Space. Sentinel-2 L2A
   (wolkenfreier Monatsmedian) und Sentinel-1 VV/VH; Ausschlussmasken serverseitig (ESA WorldCover, CORINE), OSM-Gebäude und bekannte
   Anlagen als hochgeladene Vektoren. Ausgangspunkt Eurac `clausmichele/openEO_photovoltaic` (Modell und Bänder werden in Schritt 3
   geprüft). Herunter lädt der PC nur das Wahrscheinlichkeitsraster (10 m, uint8); Polygone ab 0,5 ha entstehen lokal.
   Zugangsdaten nur über Umgebungsvariablen (`OPENEO_*`).
3. **Verdachtsliste:** Polygon, Wahrscheinlichkeit, `first_seen` (erster Monat mit PV-Signatur), Status `im_bau` (NDVI-Einbruch/Rodung
   und Strukturänderung ohne volle PV-Signatur) oder `in_betrieb`.
4. **Feinanalyse** (`ground/fine.py`, Auftrag `ground_fine`): Orthofoto-Ausschnitt mit 500 m Puffer per WMS/WMTS/WCS aus
   `ground_sources.yml` (Format = Plugin-Art `imagery`, `docs/plugins.md`). Zaunfläche und Modulfläche: zum Start ohne neues Modell über
   die FFT-Reihenanalyse aus `typology.py` (Reihenrichtung, Reihenabstand, Bedeckungsgrad); ein gelerntes Modell folgt aus geprüften
   Fällen. **Altersregel:** Orthofoto-Aufnahmedatum vor `first_seen` → nicht segmentieren, S2-Polygon gilt vorläufig, `wartet_auf =
   befliegung`; bei jedem Lauf wird das Aufnahmedatum neu geprüft. Ohne Aufnahmedatum gilt das Bild als älter.
5. **Typ und Leistung** (`ground/capacity.py`):

| Typ (`typ`) | Reihen | Merkmal |
|---|---|---|
| `fest_sued` | Ost-West | einzelne geneigte Reihen, Schattenstreifen nach Norden |
| `fest_ost_west` | Nord-Süd | Doppelreihen (zwei Helligkeiten je Reihe, `two_level`), sehr dicht |
| `tracker` | Nord-Süd | einzelne Reihen, eine Helligkeit, Abstand dazwischen |
| `agri_vertikal` | Nord-Süd | dünne Linien, breite Zwischenräume |
| `floating` | – | auf der Wassermaske |
| `unbekannt` | – | keine Reihen erkennbar (Bild zu alt/unscharf) |

   Grenzen für Reihenabstand und -breite werden an den bekannten Anlagen des Ausgangsbestands gemessen und in `ground_types.yml`
   gespeichert (änderbar), nicht fest eingetragen.

   - Baujahr aus `first_seen` bzw. der S2-Zeitreihe → Modulwirkungsgrad (`efficiency_by_year` wie bei Dächern).
   - DC = Modulfläche × Wp/m² nach Baujahr (`methode = modulflaeche`); ersatzweise Zaunfläche × Leistungsdichte nach Typ und Baujahr
     (`methode = zaunflaeche`), Tabelle `ground_density.yml`, Startwerte in MWp/ha: fest ab 2022 0,8–1,0; fest 2012–2018 0,5–0,7;
     Tracker 0,5–0,7; Agri-PV vertikal 0,3–0,4 (Jahre dazwischen linear).
   - Immer Spanne `kwp_min`/`kwp_max` und `methode`. AC = DC ÷ 1,4 bis DC ÷ 1,2. Ertrag kWh/kWp über PVGIS (Zwischenspeicher je 0,1°).
   - **Register** (geändert 18.11.2026, Christoph): Die Leistungsdichte darf mit Registern kalibriert werden, mit Schutz vor
     Fehleinträgen: nur eindeutige Zuordnungen (Einheit in genau einer Fläche bzw. einzige Fläche ≤ 100 m), Plausibilität
     0,2–2,0 MWp/ha, Median und Quartile je Baujahresklasse (mindestens 5 Paare), Testmenge räumlich zurückgehalten, Start- und
     kalibrierte Werte im Vergleich; eingeschaltet erst nach Sicht der Testzahlen (`ground_calibration`), Methode
     `zaunflaeche_kalibriert`. Deutschland: MaStR-Gesamtexport (Feld `ArtDerSolaranlage` = 852 Freiflächensolaranlage; Brutto =
     DC, Netto = AC), zweites Pilotgebiet Brandenburg. Polen: MIOZE-Register der URE (api.ure.gov.pl, Anlagen bis 1 MW, Ort und
     AC-Leistung, keine Koordinaten) nur zur **Eingrenzung der Suchregionen** (Orte mit Registerleistung ohne bekannte Fläche);
     in Polen ist die installierte PV-Leistung die Wechselrichterleistung (AC).

## Datenmodell

PostGIS auf dem PV-Mapper-PC (Docker-Container `pvmapper-db`, PostgreSQL 16 + PostGIS 3) und in PV-Map, **gleiche Tabellendefinition**.
So laufen Überlappung, Dubletten und Puffer als SQL, und die Übergabe ist ein Zeilenabgleich.

| Spalte | Typ | Bedeutung |
|---|---|---|
| `id` | text PK | `ff-<land>-<n>` (stabil, vergibt PV-Mapper) |
| `land` | char(2) | ISO-Ländercode |
| `geom_zaun` | geometry(MultiPolygon,4326) | Umzäunungsfläche bzw. S2-Polygon |
| `geom_module` | geometry(MultiPolygon,4326) null | Modulfläche (Reihen), nur nach Feinanalyse |
| `flaeche_zaun_m2`, `flaeche_module_m2` | real | aus den Geometrien (flächentreu berechnet) |
| `first_seen` | date | Monat des ersten Auftauchens (1. des Monats) |
| `status` | text | `im_bau` \| `in_betrieb` |
| `typ` | text | siehe Tabelle oben |
| `baujahr_schaetzung` | smallint | |
| `kwp_min`, `kwp_max` | real | DC-Spanne |
| `ac_kw_min`, `ac_kw_max` | real | AC-Spanne |
| `ertrag_kwh_kwp` | real | PVGIS |
| `methode` | text | `modulflaeche` \| `zaunflaeche` \| `s2_polygon` |
| `orthofoto_quelle` | text | Schlüssel aus `ground_sources.yml` |
| `orthofoto_datum` | date null | Aufnahmedatum des verwendeten Ausschnitts |
| `orthofoto_lizenz`, `orthofoto_nennung` | text | Lizenz und Quellenangabe, in der App angezeigt |
| `wartet_auf` | text null | `befliegung` (Altersregel) |
| `konfidenz` | real 0–1 | |
| `pruefstatus` | text | `kandidat` \| `manuell_geprueft` \| `gegenprobe_bestaetigt` \| `verworfen` |
| `verwurf_grund` | text null | z. B. `gebaeude` (> 20 % Überlappung), `zu_klein`, `manuell` |
| `quellen` | jsonb | `{"osm": ["way/…"], "tzsam": ["…"], "s2": "2026-10"}` |
| `zeitreihe` | jsonb | Monatswerte (Wahrscheinlichkeit, NDVI) für die Prüfliste |
| `historie` | jsonb | Prüfschritte: wer/was, wann, alter → neuer Wert |
| `updated_at` | timestamptz | für die Übergabe |

Indizes: GiST auf beiden Geometrien, `(land, pruefstatus)`. In PV-Map kommt die Tabelle mit eigener Migration (Entwurf vor dem PR,
Schritt 7) und einer Ebene `pvmapper_ground` (Art `ground`) nur für Sichtbarkeit, Kacheln und den Eintrag unter Datenquellen – **keine**
Zuordnung zu Gebäuden, kein `pv_merge`.

## Web-App (PV-Mapper)

Eigener Bereich „Freiflächen“ mit eigener Kartenebene (getrennt von Gebäuden). Prüfliste: Sentinel-Zeitreihe und Orthofoto-Ausschnitt
nebeneinander, Buttons „bestätigen“, „verwerfen“, „Typ korrigieren“; Filter Land, Status, Größe, Konfidenz; Lizenz und Quellenangabe am
Bild.

## Übergabe an PV-Map

Über den nächtlichen Abgleich: nach den Gebäuden `POST /api/detect/ground` (API-Token) mit allen seit dem letzten Lauf geänderten Zeilen.
Datenquelle „PV-Mapper Freiflächen“, unter Datenquellen ein- und ausblendbar (Standard: nur Owner). Als Anlage zählt nur
`manuell_geprueft` oder `gegenprobe_bestaetigt`; Kandidaten gehen nur mit Status-Kennzeichnung mit, `verworfen` nur als Rücknahme.

## Pilot und Abnahme

- Pilotregion konfigurierbar (`ground_pilot`); Start Polen, Woiwodschaft Wielkopolskie, dazu Brandenburg (Kalibrierung mit MaStR), dann ganz Tschechien.
- Hold-out räumlich (10-km-Blöcke), nicht zufällig.
- Trefferquote für Anlagen ab 1 ha ≥ 90 %.
- Fehlalarme: gegen den Bestand **und** eine Stichprobe von 100 Verdachtsfällen, von Hand in der Prüfliste geprüft (der Bestand ist
  unvollständig, ein Abgleich nur gegen ihn überschätzt die Fehlalarme). Ziel < 30 %.
- Medianer Leistungsfehler gegen Anlagen mit belegter Leistung (Auktions- oder Betreiberangaben; nicht Register).
- Tests: Gebäude-Überlappung (19 / 21 / 100 %), Altersregel (Bild älter, gleich alt, neuer, ohne Datum), Dubletten, Dichtetabelle.

## Rahmen

- Läuft auf dem PV-Mapper-PC; Sentinel-Rechnen bleibt in openEO.
- Drosselung je Quelle (`concurrency`, `min_interval_s`), Wiederaufnahme nach Abbruch über das Auftragssystem.
- Neue Abhängigkeiten in PV-Mapper: `openeo` (Apache-2.0), `rasterio`, `psycopg` (PostGIS), `pyogrio` (TZ-SAM-GeoPackage).

## Offene Entscheidungen

1. **TZ-SAM-Lizenz (CC BY-NC 4.0)** – entschieden 18.11.2026: TZ-SAM dient in PV-Mapper für Training und Test und darf an PV-Map
   gehen, solange PV-Map es nur für Test und Training nutzt. Jede Zeile mit TZ-SAM-Anteil trägt `quellen.lizenz_nc = true`; PV-Map zeigt
   sie nur dem Owner. Vor kommerzieller Nutzung wird die kommerzielle Lizenz bei TransitionZero angefragt; bis dahin bleiben diese Zeilen
   aus allen bezahlten Funktionen.
2. **Copernicus-Data-Space-Konto:** legt Christoph an; Zugangsdaten als Umgebungsvariablen auf dem PC.
3. **Belegte Leistungen für die Abnahme:** OSM mit Betreiberangabe (`plant:output:electricity`) plus Liste von Christoph.
4. **Zugriff auf den PC** – entschieden 18.11.2026: SSH über die Linux-VM der Claude-App auf dem Mac (Ordner `pv-map-ops`), Nutzer
   `claude` ohne Passwort und ohne allgemeines sudo, nur Schlüssel aus dem Heimnetz; erlaubt sind per sudo nur `pvmapper-update` und der
   Neustart des Dienstes.
5. **PostGIS auf dem PC** – entschieden 18.11.2026: kommt mit `pvmapper-update`.
