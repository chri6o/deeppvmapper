# Anlagentypologie – Entwurf und Stand

Ziel: Für jede erkannte Dach-PV-Anlage Bauform, Ausrichtung, Neigungsklasse und Belegungsgrad bestimmen und daraus eine
Leistung mit Spanne schätzen. Bisher galt: Modulfläche × 0,20 kWp/m².

## Was umgesetzt ist (PV-Mapper)

| Merkmal | Herkunft | Verlässlichkeit |
|---|---|---|
| Anlage, Umriss, Fläche | Maske; Felder mit < 3 m Abstand gehören zu einer Anlage | gut, so gut wie die Maske |
| Bauform `flush` / `tilted_mono` / `east_west` | Reihenmuster (2D-FFT der Maske und des Graubilds), Deckung, zwei Helligkeitsstufen je Periode | Pilot NRW: plausibel auf 7 von 8 Testgebäuden; ein Glasdach (Fehlerkennung) wurde als Ost-West eingestuft |
| Ausrichtung | Reihennormale; Süd-Reihen über den Schattenstreifen; Ost-West: beide Normalen | Ost-West gut; flächig auf Steildach: noch nicht (braucht Dachflächen) |
| Neigungsklasse | Standard je Bauform (Konfidenz 0,4–0,8) | grob; aus dem Luftbild nicht messbar |
| Belegungsgrad | gemessen aus Reihenabstand und Deckung, sonst Standard je Bauform | gemessen ± 0,05, Standard mit Spanne aus `typology.yml` |
| Leistung | Umriss × Belegungsgrad / cos(Neigung) × Wirkungsgrad (Luftbildjahr) | Spanne `kwp_low`–`kwp_high`; Kalibrierung gegen MaStR steht aus |
| integriert, Befestigung, Unterkonstruktion, Technik | – | immer `unknown` |

Ausgaben: `installations.geojson` (je Anlage Umriss, Merkmale, Konfidenzen, OSM-Tags, MaStR-Felder), Spalten
`kwp_low`, `kwp_high`, `layout`, `kwp_method` in `buildings.csv`. Abschaltbar (`typology: false`).

## Nächste Schritte

1. **Kalibrierung:** einen Nachtlauf NRW mit `pv_kwp` aus PV-Map, dann `python -m pvmapper calibrate <job>`. Erwartung:
   Belegungsgrade je Bauform aus mehreren hundert Gebäuden; Fehler Typologie gegen alte Flächen-Schätzung.
2. **Stichprobe mit Handlabels** (Bauform, Ausrichtung) für eine echte Trefferquote – 100 Gebäude NRW, 100 NL.
3. **Dachflächen für Steildächer** (Ausrichtung flächiger Anlagen, geeignete Dachfläche für Gebäude ohne PV): eigener Schritt.

## Vorschlag für PV-Map (nicht umgesetzt – wartet auf Freigabe)

Das bestehende Format `/detect/results` bleibt gültig. Neu und optional je Gebäude:

```json
{"id": "…", "result": "pv", "kwp": 48.0, "kwp_low": 40.0, "kwp_high": 57.0, "layout": "tilted_mono",
 "installations": [{"installation_id": "…#i0", "geometry": {"type": "Polygon", "…": "…"},
   "layout": "tilted_mono", "azimuth_deg": 178, "azimuth2_deg": null, "tilt_class": "5-20",
   "fill_factor": 0.56, "fill_factor_source": "derived_pitch", "outline_area_m2": 398,
   "kwp_est": 48.0, "kwp_low": 40.0, "kwp_high": 57.0, "confidence": {"layout": 0.75, "…": 0.4},
   "imagery_year": 2024, "review_status": "unreviewed"}]}
```

Migration `137_pv_installation.sql` (Entwurf):

```sql
ALTER TABLE detect_scan ADD COLUMN IF NOT EXISTS kwp_low real, ADD COLUMN IF NOT EXISTS kwp_high real,
                        ADD COLUMN IF NOT EXISTS layout text;
CREATE TABLE IF NOT EXISTS pv_installation (
  installation_id text PRIMARY KEY,
  building_id text NOT NULL REFERENCES building ON DELETE CASCADE,
  source text NOT NULL DEFAULT 'pvmapper',
  geom geometry(Polygon, 4326),
  layout text, azimuth_deg real, azimuth2_deg real, tilt_class text,
  fill_factor real, fill_factor_source text, outline_area_m2 real,
  kwp_est real, kwp_low real, kwp_high real,
  confidence jsonb NOT NULL DEFAULT '{}', attrs jsonb NOT NULL DEFAULT '{}',
  imagery_year smallint, review_status text NOT NULL DEFAULT 'unreviewed',
  updated_at timestamptz NOT NULL DEFAULT now());
CREATE INDEX IF NOT EXISTS pv_installation_building ON pv_installation(building_id);
-- RLS like detect_scan: owner only until the layer visibility allows more
```

Eine neue Prüfung ersetzt alle Anlagen des Gebäudes (Löschen + Einfügen in einer Transaktion). Die Gebäudetabelle selbst
bleibt unverändert; `building.kwp` wird nicht überschrieben. Im Steckbrief würde die Spanne neben der Registerleistung stehen.
Auf PV-Mapper-Seite genügt dann `typology_push: true`.
