# Gothenburg labels: files and tools

Viewport: `gothenburg_viewport.geojson` on `main` (11.925-12.0091 E, 57.6848-57.7297 N, grid 661 x 353).
Classes: FOREST, BUILDINGS, STREET, RIVER, FOOTBALL FIELD. North of the line in
`region_elias-christopher.geojson`: Elias + Christopher; south: Alice + Isabel.

## Files in `data/`

| File | Content |
|---|---|
| `alice_2026-10-12_hand-drawn.geojson` | the 950 hand-drawn polygons of Alice's export of 12 Oct (`manual-labels-Gteborg (1).geojson` on `main`, type `polygon` only) |
| `region_elias-christopher.geojson` | outline of the north region (from `gothenburg.zip` on `main`) |
| `osm_draft_north.geojson`, `osm_draft_south.geojson` | draft labels from OpenStreetMap on the pixels that have no hand-drawn label yet (type `osm`) |
| `osm_draft.npz`, `overview_alice_vs_osm-draft.png` | class shares per pixel and an overview map of hand-drawn labels and draft |

## Labelling in TEE

1. Create the viewport from `gothenburg_viewport.geojson` (viewport selector, Import). If a viewport called
   "Gteborg" exists already, change `"name"` in the file first.
2. Viewer -> Labelling -> Manual Label -> Import -> From file...: `region_elias-christopher.geojson` and, to see
   what is done, `alice_2026-10-12_hand-drawn.geojson`.
3. Draw with the class names above. Export -> **GeoJSON**, "Classified pixels" unticked.

Do not import the OSM draft or Alice's full export (with the promoted forest cluster) into the TEE session you
label in: TEE's import turns every polygon into type `polygon`, so after the next export they could no longer be
told apart from hand-drawn ones. Use the draft only in `merge_labels.py`.

## Tools

- `tee_export_to_geojson.py EXPORT OUT.geojson [--types polygon] [--regions-out R.geojson]`: reads a TEE export
  (shapefile ZIP with all layers, or GeoJSON), keeps the chosen label types, and writes a GeoJSON that TEE imports.
  Needed for ZIPs written before upstream PR #27 (TEE v1.9.0-54).
- `osm_prelabel.py --alice EXPORT [EXPORT ...] --out-dir data`: draft labels from OpenStreetMap (buildings, water,
  wood / forest, soccer pitches, roads buffered to their width; a pixel gets a class when it covers >= 50 % of it),
  compared with the hand-drawn polygons. On 12 Oct: 96.2 % agreement on the 40,897 pixels both label.
- `../spatial_hdbscan/merge_labels.py OUT.geojson EXPORT ... --types polygon,osm`: combines exports; the later file
  wins, so put the OSM draft first and the hand-drawn exports after it.

The OSM draft contains data (c) OpenStreetMap contributors, available under the Open Database License (ODbL),
https://www.openstreetmap.org/copyright.
