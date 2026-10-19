"""Combine the label exports of several people into one ground truth and check them
(Christopher, 12 Oct 2026; the step that was still open in the meeting of 12 Oct).

    ~/tee-local/venv/bin/python merge_labels.py OUT.geojson EXPORT [EXPORT ...]
        [--metadata ~/tee_data/vectors/Gteborg/2024/metadata.json] [--ignore "NAME,NAME"]

EXPORT   a TEE manual-label export: the shapefile ZIP (as in gothenburg.zip) or a GeoJSON file.
OUT      the combined class map of the --metadata grid as GeoJSON (polygons along the pixel edges,
         properties name / color / type "merged"), usable as GBG_LABELS=OUT.geojson for
         run_final.py / run_spectral.py on the same grid, and as OUT.npz (class raster).

Class names are compared in upper case without surrounding blanks. Features whose name contains a
team member's name (the regions drawn to split the work, e.g. "ELIAS and CHRISTOFER") or is listed
in --ignore are not classes; they are reported as regions with the share that is labelled inside.
Where two exports label the same pixel differently, the later export on the command line wins and
the conflict is reported. Hence the OpenStreetMap draft goes first and the hand-drawn exports after it.

Special names (15 Oct, for correcting the OSM draft in TEE):
  NONE        erases the labels of the earlier exports under it (for pixels that fit none of the
              classes); the same export's own class labels still apply on top.
  REF ...     reference layers (e.g. the OSM draft imported as "REF BUILDINGS" to see it in TEE)
              are skipped.

--types polygon   keep only labels of these TEE types. "polygon" = drawn by hand; "similarity" = a
                  promoted auto-label (k-means) cluster and "classified" = TEE's nearest-centroid
                  classification, both made from the embeddings that the clustering is scored on.
                  "osm" = draft from osm_prelabel.py.
"""
import argparse, json, re, sys, warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gothenburg_labels"))
import tee_export_to_geojson as tee  # noqa: E402  (reads the .shp records without GDAL's ring guessing)

warnings.filterwarnings("ignore")
TEAM = re.compile(r"ELIAS|CHRISTO|ALICE|ISABEL", re.I)
FROM_EMBEDDINGS = {"similarity", "classified"}
ERASE = {"NONE", "NO LABEL", "UNLABELLED", "ERASE"}
is_ref = lambda n: n.startswith("REF ")


def read_export(path, types=()):
    """Polygon features of one export as a GeoDataFrame with 'name' (normalised) and 'type' (EPSG:4326)."""
    import geopandas as gpd
    from shapely.geometry import Polygon
    rows = []
    for l in tee.read_export(path):
        if l["kind"] != "polygon":
            continue
        geom = tee.rings_to_geometry(l["geom"])
        rows.append({"name": l["props"]["name"].upper(), "type": l["props"].get("type") or "polygon",
                     "color": l["props"].get("color") or "", "geometry": geom})
    if not rows:
        raise SystemExit(f"{path}: no polygons")
    g = gpd.GeoDataFrame(rows, crs=4326)
    if types:
        g = g[g["type"].isin(types) | g["name"].str.contains(TEAM) | g["name"].isin(ERASE)]
    return g[["name", "type", "color", "geometry"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out"); ap.add_argument("exports", nargs="+")
    ap.add_argument("--metadata", default=str(Path.home() / "tee_data/vectors/Gteborg/2024/metadata.json"))
    ap.add_argument("--ignore", default="")
    ap.add_argument("--types", default="")
    a = ap.parse_args()
    from affine import Affine
    from rasterio.features import rasterize
    meta = json.loads(Path(a.metadata).read_text())
    gt, shape = meta["geotransform"], (meta["mosaic_height"], meta["mosaic_width"])
    T = Affine(gt["a"], 0.0, gt["c"], 0.0, gt["e"], gt["f"])
    ignore = {n.strip().upper() for n in a.ignore.split(",") if n.strip()}
    is_region = lambda n: n in ignore or bool(TEAM.search(n))

    types = tuple(t.strip() for t in a.types.split(",") if t.strip())
    sources = [(Path(p).name, read_export(p, types)) for p in a.exports]
    for name, g in sources:
        if g["name"].map(is_ref).any():
            print(f"{name}: skipped {int(g['name'].map(is_ref).sum())} reference polygons (REF ...)")
    sources = [(name, g[~g["name"].map(is_ref)]) for name, g in sources]
    classes = sorted({n for _, g in sources for n in g["name"].unique() if not is_region(n) and n not in ERASE})
    colors = {}
    for _, g in sources:
        for n, c in zip(g["name"], g["color"]):
            if c and n not in colors:
                colors[n] = c
    print(f"viewport grid {shape[1]} x {shape[0]} pixels; classes: {classes}")
    combined = np.zeros(shape, np.int16)
    rasters, regions = [], []
    for name, g in sources:
        reg = g[g["name"].map(is_region)]
        regions += [(f"{n} ({name})", geom) for n, geom in zip(reg["name"], reg.geometry)]
        g = g[~g["name"].map(is_region)]
        er = g[g["name"].isin(ERASE)]
        if len(er):
            m = rasterize([(geom, 1) for geom in er.geometry], out_shape=shape, transform=T, fill=0, dtype="uint8").astype(bool)
            print(f"\n{name}: NONE polygons erase {int((m & (combined > 0)).sum())} labelled pixels of the earlier exports")
            combined[m] = 0
            g = g[~g["name"].isin(ERASE)]
        r = rasterize([(geom, classes.index(n) + 1) for n, geom in zip(g["name"], g.geometry)], out_shape=shape, transform=T,
                      fill=0, dtype="int16") if len(g) else np.zeros(shape, np.int16)
        rasters.append(r)
        print(f"\n{name}: {len(g)} polygons, {int((r > 0).sum())} labelled pixels ({100 * (r > 0).mean():.1f} % of the viewport)")
        emb = g[g["type"].isin(FROM_EMBEDDINGS)]
        if len(emb):
            m = rasterize([(geom, 1) for geom in emb.geometry], out_shape=shape, transform=T, fill=0, dtype="uint8")
            print(f"    NOTE: {int(m.sum())} of these pixels come from TEE types {sorted(emb['type'].unique())} "
                  f"(made from the embeddings, not an independent ground truth); --types polygon leaves them out")
        for i, c in enumerate(classes, 1):
            if (r == i).any():
                print(f"    {c:16s} {int((r == i).sum()):7d} px")
        both = (combined > 0) & (r > 0)
        if both.any():
            conflict = both & (combined != r)
            print(f"    overlaps earlier exports on {int(both.sum())} px; same class on {int((both & ~conflict).sum())}, DIFFERENT class on {int(conflict.sum())}")
            pairs, counts = np.unique(np.stack([combined[conflict], r[conflict]], 1), axis=0, return_counts=True)
            for (x, y), n in sorted(zip(map(tuple, pairs), counts), key=lambda t: -t[1])[:8]:
                print(f"        {classes[x - 1]} -> {classes[y - 1]}: {n} px (this export wins)")
        combined = np.where(r > 0, r, combined)
    print(f"\ncombined: {int((combined > 0).sum())} labelled pixels = {100 * (combined > 0).mean():.1f} % of the viewport")
    for i, c in enumerate(classes, 1):
        print(f"    {c:16s} {int((combined == i).sum()):7d} px  {100 * (combined == i).mean():5.1f} %")
    for label, geom in regions:
        m = rasterize([(geom, 1)], out_shape=shape, transform=T, fill=0, dtype="uint8").astype(bool)
        if m.any():
            print(f"region {label}: {100 * m.mean():.1f} % of the viewport, {100 * (combined[m] > 0).mean():.1f} % of it labelled")
    out = Path(a.out)
    feats = [{"type": "Feature", "properties": {"name": c, "color": colors.get(c, ""), "type": "merged"},
              "geometry": {"type": "Polygon", "coordinates": [ring]}}
             for i, c in enumerate(classes, 1) for ring in tee.mask_polygons(combined == i, gt)]
    out.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    np.savez_compressed(out.with_suffix(".npz"), cls=combined, classes=np.array(classes))
    print(f"\nwrote {out} ({len(feats)} polygons along the pixel edges) and {out.with_suffix('.npz').name}")


if __name__ == "__main__":
    main()
