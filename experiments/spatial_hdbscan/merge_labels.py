"""Combine the label exports of several people into one ground truth and check them
(Christopher, 12 Oct 2026; the step that was still open in the meeting of 12 Oct).

    ~/tee-local/venv/bin/python merge_labels.py OUT.geojson EXPORT [EXPORT ...]
        [--metadata ~/tee_data/vectors/Gteborg/2024/metadata.json] [--ignore "NAME,NAME"]

EXPORT   a TEE manual-label export: the shapefile ZIP (as in gothenburg.zip) or a GeoJSON file.
OUT      combined GeoJSON with one property "name" (class, upper case), usable as
         GBG_LABELS=OUT.geojson for run_final.py / run_spectral.py. Also writes OUT.npz with the
         class raster on the viewport grid.

Class names are compared in upper case without surrounding blanks. Features whose name contains a
team member's name (the regions drawn to split the work, e.g. "ELIAS and CHRISTOFER") or is listed
in --ignore are not classes; they are reported as regions with the share that is labelled inside.
Where two exports label the same pixel differently, the later export on the command line wins and
the conflict is reported.
"""
import argparse, json, re, sys, warnings, zipfile
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
TEAM = re.compile(r"ELIAS|CHRISTO|ALICE|ISABEL", re.I)


def read_export(path):
    """All polygon features of one export as a GeoDataFrame with a normalised 'name' column (EPSG:4326)."""
    import geopandas as gpd
    import pandas as pd
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            layers = [n for n in z.namelist() if n.lower().endswith(".shp")]
        parts = [gpd.read_file(f"zip://{path}!{n}") for n in layers]
        g = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
    else:
        g = gpd.read_file(path)
    key = next((c for c in ("name", "label_name", "class", "label") if c in g.columns), None)
    if key is None:
        raise SystemExit(f"{path}: no name / label_name column ({list(g.columns)})")
    g = g[g.geometry.notna() & g.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    g["name"] = g[key].astype(str).str.strip().str.upper()
    return g.to_crs(4326)[["name", "geometry"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out"); ap.add_argument("exports", nargs="+")
    ap.add_argument("--metadata", default=str(Path.home() / "tee_data/vectors/Gteborg/2024/metadata.json"))
    ap.add_argument("--ignore", default="")
    a = ap.parse_args()
    import geopandas as gpd
    import pandas as pd
    from affine import Affine
    from rasterio.features import rasterize
    meta = json.loads(Path(a.metadata).read_text())
    gt, shape = meta["geotransform"], (meta["mosaic_height"], meta["mosaic_width"])
    T = Affine(gt["a"], 0.0, gt["c"], 0.0, gt["e"], gt["f"])
    ignore = {n.strip().upper() for n in a.ignore.split(",") if n.strip()}
    is_region = lambda n: n in ignore or bool(TEAM.search(n))

    sources = [(Path(p).name, read_export(p)) for p in a.exports]
    classes = sorted({n for _, g in sources for n in g["name"].unique() if not is_region(n)})
    print(f"viewport grid {shape[1]} x {shape[0]} pixels; classes: {classes}")
    combined = np.zeros(shape, np.int16)
    rasters, regions = [], []
    for name, g in sources:
        reg = g[g["name"].map(is_region)]
        regions += [(f"{n} ({name})", geom) for n, geom in zip(reg["name"], reg.geometry)]
        g = g[~g["name"].map(is_region)]
        r = rasterize([(geom, classes.index(n) + 1) for n, geom in zip(g["name"], g.geometry)], out_shape=shape, transform=T,
                      fill=0, dtype="int16") if len(g) else np.zeros(shape, np.int16)
        rasters.append(r)
        print(f"\n{name}: {len(g)} polygons, {int((r > 0).sum())} labelled pixels ({100 * (r > 0).mean():.1f} % of the viewport)")
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
    allg = gpd.GeoDataFrame(pd.concat([g[~g["name"].map(is_region)] for _, g in sources], ignore_index=True), crs=4326)
    allg.to_file(out, driver="GeoJSON")
    np.savez_compressed(out.with_suffix(".npz"), cls=combined, classes=np.array(classes))
    print(f"\nwrote {out} ({len(allg)} polygons) and {out.with_suffix('.npz').name}")


if __name__ == "__main__":
    main()
