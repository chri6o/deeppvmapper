# Task: Add rooftop PV installation typology and fill factor to PV-Mapper

## Context

PV-Mapper (fork `chri6o/deeppvmapper`, based on DeepPVMapper by gabrielkasmi) detects PV installations on buildings from aerial orthoimagery. Pilot countries: DE, NL, AT. Building lists come from PV-Map (OSM building footprints, public buildings). Results flow nightly into PV-Map; only hand-checked or cross-validated detections count as PV. PV-Map compares detections against the German Marktstammdatenregister (MaStR) and municipal reports.

Goal of this task: for every detected installation, also determine **what kind of installation it is**, its **outline**, its **fill factor**, and from that a **capacity estimate**. Use terms that match standards and registers, so the results can be compared with MaStR and OSM.

Before changing anything, read the repo and summarise how detection, segmentation and characterisation (azimuth, tilt, surface, kWp) currently work, and where the new attributes fit. Propose the change plan, then implement.

## 1. Classification schema

Each installation gets these attributes. They are independent of each other; do not merge them into one flat type list.

| Attribute | Values | Terminology source |
|---|---|---|
| `integration` | `BAPV`, `BIPV_A`, `BIPV_B`, `unknown` | EN 50583 / IEC 63092 (A = sloped, roof-integrated, not accessible from inside; B = sloped, roof-integrated, accessible from inside / skylight). BAPV = building-attached, modules are not a construction product. |
| `roof_shape` | OSM `roof:shape` values (`flat`, `gabled`, `hipped`, `skillion`, `sawtooth`, `round`, …) | OSM. Take from OSM building data where tagged; otherwise estimate. |
| `mounting_pitched` (pitched roofs only) | `on_roof` (Aufdach), `in_roof` (Indach), `solar_tiles`, `unknown` | ZVDH Merkblatt Solartechnik für Dach und Wand; Swissolar |
| `securing_flat` (flat roofs only) | `laid_ballasted` (aufgelegt, ballastiert), `laid_aerodynamic` (aufgelegt, aerodynamisch), `anchored` (aufgeständert = fixed to the load-bearing structure via bases through the membrane), `unknown` | Merkblatt Solar auf Flachdächern, Der dichte Bau (update 2026). Note: in this Merkblatt "aufgeständert" means *anchored to the structure*, not "tilted". Do not use "aufgeständert" for tilt. |
| `carrier` (optional, flat roofs) | `rail_rack`, `tray` (Wannensystem / ballast tray, e.g. Renusol ConSole type), `green_roof`, `none`, `unknown` | Own extension, not standardised |
| `layout` | `flush` (parallel to roof surface), `tilted_mono` (single orientation, rows), `east_west` (back-to-back pairs), `vertical_bifacial`, `mixed`, `unknown` | Industry usage |
| `row_alignment` (flat roofs) | `aligned` (parallel to building edge), `rotated` (turned towards a compass direction), `n/a` | Own extension |
| `module_format` | `portrait`, `landscape`, `mixed`, `unknown` | Industry usage, not standardised |
| `azimuth_deg` | 0–359 (0 = north, clockwise). For `east_west` store both values (`azimuth_deg`, `azimuth2_deg`). | OSM `generator:orientation`; MaStR registers East-West as two sub-units |
| `tilt_class` | MaStR classes: `<5` (horizontal), `5-20`, `21-40`, `41-60`, `61-89`, `90` (vertical), `non_uniform` | MaStR registration help (as of 05.12.2025). Also keep `tilt_deg` if estimated. For comparison with older MaStR entries map to `<20 / 20-40 / 40-60 / >60`. |
| `module_count` | integer or null | OSM `generator:solar:modules` |

Optional technology flag if detectable: `standard`, `bifacial`, `thin_film_flexible`, `pvt`, `unknown`.

## 2. Visual cues (for model features, rules or reviewer guidance)

- **Pitched, on-roof:** homogeneous module field parallel to the roof surface, slight offset/shadow at the field edge, roof covering visible around it.
- **In-roof / solar tiles:** flush with the roof, no edge shadow. Hard to separate from on-roof; solar tiles hard to separate from ordinary roofing. Mark `unknown` rather than guess.
- **East-West:** dense zig-zag/tent pattern, pairs back to back, little or no gap between pairs; maintenance aisles possible.
- **South tilted rows (rail rack or tray):** parallel rows with visible shadow strips in between.
- **Tray system (Wannensystem):** each module on its own carrier, small gaps between modules within a row, gravel or stones visible in the trays, no continuous rails.
- **Flush on flat roof:** homogeneous surface, no row shadows; often lightweight industrial roofs.
- **Vertical bifacial:** thin lines with wide spacing; often on green roofs.
- **Rotated rows:** rows at an angle to the building edges.

## 3. Outline and fill factor

Store two levels separately:

1. **Installation fill factor** (Ground Coverage Ratio, GCR) = module area / area of the installation outline polygon.
2. **Roof occupancy** = installation outline area / usable roof area (for remaining potential).

Determine the installation fill factor in this order of preference and record the method in `fill_factor_source`:

- `counted`: module_count × standard module footprint (projected) / outline area.
- `derived_pitch`: from measured geometry:
  - `f = f_row × f_cross`
  - `f_row = L · cos(β) / p` (L = module length in tilt direction, β = tilt, p = row pitch front edge to front edge, measured in plan view)
  - `f_cross = B / (B + g)` (B = module width, g = gap between modules within a row)
  - For East-West, treat each back-to-back pair as one unit; p = pair-to-pair distance including aisles.
  - The cos(β) term gives the projected (plan view) value. For real module area, omit it.
- `type_default`: if nothing can be measured, use defaults by layout (plan view, rule-of-thumb values, to be calibrated against verified installations):

| Layout | Default f |
|---|---|
| Pitched, flush | 0.93 (range 0.90–0.97) |
| Flat, flush | 0.92 (0.90–0.95) |
| East-West, no aisles | 0.85 (0.80–0.90) |
| East-West, with aisles | 0.72 (0.65–0.80) |
| South tilted / tray | 0.50 (0.40–0.60, depends on tilt and latitude) |
| Vertical bifacial | 0.18 (0.10–0.25) |

Make the defaults configurable and add a calibration script that fits them from installations that are both detected and matched to MaStR.

## 4. Capacity estimate

`kWp_est = outline_area_m2 × f_real × module_efficiency × 1 kW/m²`

- Use real module area (without cos β), not the projected value.
- `module_efficiency` configurable, default by installation year if known, else 0.21.
- Output a range (low/high) based on the fill factor range and an efficiency range, plus the point estimate.
- Example check: outline 100 m², f = 0.52, efficiency 0.22 → ~11.4 kWp.

## 5. Output and data model

Per installation, output (JSON/GeoJSON feature properties, and the corresponding DB columns for the PV-Map import):

```
installation_id, building_id, geometry (outline polygon),
integration, roof_shape, mounting_pitched, securing_flat, carrier,
layout, row_alignment, module_format, technology,
azimuth_deg, azimuth2_deg, tilt_deg, tilt_class,
module_count, outline_area_m2,
fill_factor, fill_factor_source, roof_occupancy,
kwp_est, kwp_low, kwp_high,
confidence (per attribute, 0–1), imagery_source, imagery_year,
review_status (unreviewed | confirmed_manual | confirmed_crosscheck | rejected)
```

Also provide an export mapping to OSM tags (`power=generator`, `generator:source=solar`, `generator:method=photovoltaic`, `generator:type=solar_photovoltaic_panel`, `location=roof`, `generator:orientation`, `generator:solar:modules`) and to MaStR fields (Hauptausrichtung, Neigungswinkel der Hauptausrichtung, Nebenausrichtung, Neigungswinkel der Nebenausrichtung, Anzahl der Module). Mounting type has no OSM tag; keep it in PV-Map only.

## 6. Rules

- Prefer `unknown` over a low-confidence guess; keep per-attribute confidence.
- Attributes that do not apply to the roof type are null (e.g. `securing_flat` on a pitched roof).
- One building can have several installations (multi-face pitched roofs, several fields on a flat roof). Do not merge fields with different azimuth or layout.
- Re-run classification when a new imagery year becomes available; keep history.
- Do not change the existing review logic: only hand-checked or cross-validated detections count as PV in PV-Map.

## 7. Deliverables and acceptance

1. Short design note: where in the pipeline each attribute is produced (model output, geometry post-processing, rules), with the change plan.
2. Implementation of schema, fill factor computation, capacity estimate, exports.
3. Config file for module dimensions, efficiency and fill factor defaults.
4. Tests: unit tests for the fill factor formulas (incl. the example above and an East-West case), schema validation, OSM/MaStR mapping.
5. A labelled sample set (at least 20 buildings per layout class from DE, NL, AT) and an evaluation report: per-attribute accuracy, fill factor error, kWp error against MaStR where matched.
6. Updated README section describing the attributes and their sources.

Ask before changing the PV-Map import format or the DB schema in PV-Map itself; propose a migration first.
