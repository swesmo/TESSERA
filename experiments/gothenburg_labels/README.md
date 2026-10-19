# Gothenburg labels: files and tools

Viewport for labelling: `gothenburg_viewport.geojson` on `main` (11.925-12.0091 E, 57.6848-57.7297 N, grid
661 x 353). Classes: FOREST, BUILDINGS, STREET, RIVER, FOOTBALL FIELD. North of the line in
`region_elias-christopher.geojson`: Elias + Christopher; south: Alice + Isabel.

Decision of 15 Oct (Christopher): the OpenStreetMap draft is the base layer of the ground truth; hand-drawn
labels go on top and correct it.

## Files in `data/`

| File | Content |
|---|---|
| `alice_2026-10-12_hand-drawn.geojson` | the 950 hand-drawn polygons of Alice's export of 12 Oct (`manual-labels-Gteborg (1).geojson` on `main`, type `polygon` only) |
| `region_elias-christopher.geojson` | outline of the north region (from `gothenburg.zip` on `main`) |
| `osm_draft_{north,south}.geojson` | OSM draft on Alice's grid, on the pixels without a hand-drawn label (type `osm`) |
| `osm_draft_{north,south}_REF.geojson` | the same with classes renamed `REF <class>` in lighter colours, to see the draft in TEE |
| `osm_draft.npz`, `overview_alice_vs_osm-draft.png` | class shares per pixel; overview map |
| `eval_grid_Gteborg/` | the draft on the grid of the embeddings the experiments use (old viewport `Gteborg`, 720 x 385, v1.0) and `gt_osm+hand_2026-10-15.geojson` / `.npz`: draft + Alice's hand-drawn labels, the input for `GBG_LABELS` |

OSM extract of 14 Oct 2026 (Overpass), cached in `~/tee_data/harness-cache/osm_gothenburg_union.json`.

## Correcting the draft in TEE

1. Create the viewport from `gothenburg_viewport.geojson` (viewport selector, Import). If a viewport called
   "Gteborg" exists already, change `"name"` in the file first. TEE v1.9.0-54 or later (fork `main` since 15 Oct)
   imports large files in seconds.
2. Viewer -> Labelling -> Manual Label -> Import -> From file...: `region_elias-christopher.geojson`,
   `alice_2026-10-12_hand-drawn.geojson` and `osm_draft_<your region>_REF.geojson` (one-off freeze of about a
   minute in a normal Chrome window for the 5,046 polygons of the north).
3. Compare the REF classes with the satellite image and draw only what has to change, with the real class names:
   - draft wrong (e.g. a REF BUILDINGS area that is a car park): draw the correct class over it, or `NONE` if it
     is none of the five classes; `NONE` removes the draft there.
   - draft missing (trees in parks, wide streets, pitches without OSM tag): draw the class.
4. Export -> **GeoJSON**, "Classified pixels" unticked. The export also contains the REF and Alice's polygons;
   `merge_labels.py` skips REF names, and Alice's copies are harmless as long as her own newest export comes later
   in the merge.

Do not import the plain OSM draft or Alice's full export (with the promoted forest cluster) under the real class
names: TEE's import turns every polygon into type `polygon`, so after the next export they could no longer be told
apart from hand-drawn ones.

## Building the ground truth and scoring it

    python ../spatial_hdbscan/merge_labels.py data/eval_grid_Gteborg/gt_osm+hand_<date>.geojson \
        data/eval_grid_Gteborg/osm_draft_north.geojson data/eval_grid_Gteborg/osm_draft_south.geojson \
        data/alice_2026-10-12_hand-drawn.geojson <corrections of the others ...> data/region_elias-christopher.geojson \
        --types polygon,osm
    GBG_LABELS=data/eval_grid_Gteborg/gt_osm+hand_<date>.geojson TAG=_labels_osmhand \
        python ../spatial_hdbscan/run_final.py gbg 20 all "base|over|gover"

The later file wins, so the draft goes first. On 15 Oct (draft + Alice's hand-drawn labels, 149,377 pixels =
54 % of the evaluation viewport; k = 5, 20 seeds): baseline AMI 0.399 +- 0.032 / ARI 0.346, over-cluster 32 + MRF
0.424 +- 0.001 / 0.366, stability 0.63 -> 0.95. Draft alone on the pixels without a hand label: 0.358 -> 0.371.
Hand-drawn alone: 0.467 -> 0.497. MRF smoothing helps on the hand-drawn polygons (+0.009 AMI) but not on the
draft, which keeps single buildings and streets.

## Tools

- `tee_export_to_geojson.py EXPORT OUT.geojson [--types polygon] [--regions-out R.geojson]`: reads a TEE export
  (shapefile ZIP with all layers, or GeoJSON), keeps the chosen label types, and writes a GeoJSON that TEE imports.
  Needed for ZIPs written before upstream PR #27 (TEE v1.9.0-54).
- `osm_prelabel.py --alice EXPORT [EXPORT ...] --out-dir data [--metadata GRID] [--ref]`: draft labels from
  OpenStreetMap (buildings, water, wood / forest, soccer pitches, roads buffered to their width; a pixel gets a class
  when it covers >= 50 % of it), compared with the hand-drawn polygons: 96.2 % agreement on the 40,850 pixels both
  label (Alice's grid), 95.7 % on the evaluation grid.
- `../spatial_hdbscan/merge_labels.py`: combines exports by label type (`--types`), `NONE` erases, `REF ...` is
  skipped; writes the combined class map of the `--metadata` grid (default: evaluation grid).

The OSM draft contains data (c) OpenStreetMap contributors, available under the Open Database License (ODbL),
https://www.openstreetmap.org/copyright.
