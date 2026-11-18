-- Ground-mounted PV (PV-Map docs/freiflaechen.md, #377). Same table definition as in PV-Map. Idempotent.
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS freiflaeche (
  id                  text PRIMARY KEY,                 -- ff-<cc>-<origin>-<key>, stable across imports
  land                char(2) NOT NULL,
  region              text,                             -- pilot region key (e.g. pl-wielkopolskie)
  geom_zaun           geometry(MultiPolygon,4326) NOT NULL,
  geom_module         geometry(MultiPolygon,4326),
  flaeche_zaun_m2     real,
  flaeche_module_m2   real,
  first_seen          date,
  status              text CHECK (status IN ('im_bau','in_betrieb')),
  typ                 text NOT NULL DEFAULT 'unbekannt'
                      CHECK (typ IN ('fest_sued','fest_ost_west','tracker','agri_vertikal','floating','unbekannt')),
  baujahr_schaetzung  smallint,
  kwp_min             real,
  kwp_max             real,
  ac_kw_min           real,
  ac_kw_max           real,
  ertrag_kwh_kwp      real,
  methode             text CHECK (methode IN ('modulflaeche','zaunflaeche','s2_polygon')),
  orthofoto_quelle    text,
  orthofoto_datum     date,
  orthofoto_lizenz    text,
  orthofoto_nennung   text,
  wartet_auf          text,
  konfidenz           real,
  pruefstatus         text NOT NULL DEFAULT 'kandidat'
                      CHECK (pruefstatus IN ('kandidat','manuell_geprueft','gegenprobe_bestaetigt','verworfen')),
  verwurf_grund       text,
  herkunft            text NOT NULL CHECK (herkunft IN ('bestand_osm','bestand_tzsam','suche')),
  split               text CHECK (split IN ('train','test')),   -- spatial hold-out (10 km blocks)
  quellen             jsonb NOT NULL DEFAULT '{}'::jsonb,
  zeitreihe           jsonb,
  historie            jsonb NOT NULL DEFAULT '[]'::jsonb,
  updated_at          timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS freiflaeche_zaun ON freiflaeche USING gist(geom_zaun);
CREATE INDEX IF NOT EXISTS freiflaeche_module ON freiflaeche USING gist(geom_module);
CREATE INDEX IF NOT EXISTS freiflaeche_land_pruef ON freiflaeche (land, pruefstatus);

-- source files already imported (version = Last-Modified / Zenodo record), so a re-run skips unchanged inputs
CREATE TABLE IF NOT EXISTS ground_import (
  source     text NOT NULL,            -- osm:<region> | tzsam:<record>
  version    text NOT NULL,
  rows       integer,
  imported   timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (source, version)
);

-- registers (PV-Map docs/freiflaechen.md, decision 18.11.2026): MaStR (DE, coordinates, gross = DC and net = AC) for the
-- density calibration; MIOZE (PL, place name and AC capacity) to narrow the search regions. Kept apart from freiflaeche so a
-- re-import of the baseline does not touch them.
CREATE TABLE IF NOT EXISTS ground_register (
  source      text NOT NULL,                -- mastr | mioze
  unit_id     text NOT NULL,
  region      text,
  name        text,
  ort         text,
  geom        geometry(Point,4326),         -- MaStR: unit coordinates; MIOZE: centre of the place (OSM)
  geom_kind   text,                         -- einheit | ort | mehrdeutig | fehlt
  brutto_kw   real,                         -- DC (module) where the register has it
  netto_kw    real,                         -- AC (inverters)
  start_date  date,
  attrs       jsonb NOT NULL DEFAULT '{}'::jsonb,
  imported    timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (source, unit_id)
);
CREATE INDEX IF NOT EXISTS ground_register_geom ON ground_register USING gist(geom);
CREATE TABLE IF NOT EXISTS ground_register_match (
  source   text NOT NULL,
  unit_id  text NOT NULL,
  fid      text NOT NULL,                   -- freiflaeche.id
  dist_m   real,
  PRIMARY KEY (source, unit_id)
);
ALTER TABLE freiflaeche DROP CONSTRAINT IF EXISTS freiflaeche_methode_check;
ALTER TABLE freiflaeche ADD CONSTRAINT freiflaeche_methode_check
  CHECK (methode IN ('modulflaeche','zaunflaeche','zaunflaeche_kalibriert','s2_polygon'));
