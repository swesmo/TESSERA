"""Turn a TEE manual-label export (shapefile ZIP or GeoJSON) into one GeoJSON that TEE can import
again (Christopher, 12 Oct 2026).

    ~/tee-local/venv/bin/python tee_export_to_geojson.py EXPORT.zip|EXPORT.geojson OUT.geojson
        [--metadata ~/tee_data/vectors/Gteborg_GT/2024/metadata.json]
        [--types polygon,similarity,point] [--regions-out REGIONS.geojson]

Why this is needed (TEE v1.8, public/js/labels.js; same code upstream on 12 Oct 2026):
  * "Export -> ESRI Shapefile (ZIP)" writes one layer per geometry type (POINT.shp, POLYGON.shp), but
    "Import -> From file..." reads only the first .shp in the ZIP. One pin among the labels is enough
    to make the import return that pin and silently drop every polygon.
  * The importer skips MultiPolygon features and keeps only the outer ring of a polygon. Promoted
    auto-label clusters are written with their outer ring counter-clockwise, which shapefile readers
    take for a hole, so those records come back as MultiPolygons and are dropped as well.

A GeoJSON export has neither problem, except that the holes of cluster polygons are filled on
import; it is accepted here as well, mainly for --types.

What it does: reads every layer straight from the .shp records (rings combined with the even-odd rule,
independent of their orientation), keeps hand-drawn polygons vertex for vertex, and re-draws
pixel labels (type "similarity" = promoted auto-label cluster) per class as polygons without holes
that follow the pixel edges of the viewport grid, so TEE's importer reproduces the same pixels.
Properties name / color / code / type are kept; `type` says how a label was made.

--types          keep only these label types (e.g. "polygon" for the hand-drawn ground truth)
--regions-out    features named after team members (the outlines that split the work, e.g.
                 "ELIAS and CHRISTOFER") are never written to OUT; with this option they go to
                 their own file
"""
import argparse, json, re, struct, sys, warnings, zipfile
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
TEAM = re.compile(r"ELIAS|CHRISTO|ALICE|ISABEL", re.I)


def read_shp(raw):
    """Records of one .shp as ('point', (x, y)) / ('polygon', [ring, ...]) / (None, None)."""
    pos, out = 100, []
    while pos < len(raw):
        _, clen = struct.unpack(">ii", raw[pos:pos + 8])
        body = raw[pos + 8: pos + 8 + clen * 2]
        pos += 8 + clen * 2
        st = struct.unpack("<i", body[:4])[0]
        if st == 1:
            out.append(("point", struct.unpack("<dd", body[4:20])))
        elif st == 5:
            nparts, npts = struct.unpack("<ii", body[36:44])
            parts = list(struct.unpack(f"<{nparts}i", body[44:44 + 4 * nparts])) + [npts]
            pts = np.frombuffer(body[44 + 4 * nparts: 44 + 4 * nparts + 16 * npts], dtype="<f8").reshape(-1, 2)
            out.append(("polygon", [pts[parts[i]:parts[i + 1]] for i in range(nparts)]))
        else:
            out.append((None, None))
    return out


def read_export(path):
    """All labels of an export (shapefile ZIP or GeoJSON) as dicts {props, kind, geom}, in file order.
    kind 'point': geom = (lon, lat); kind 'polygon': geom = list of rings, to be filled even-odd."""
    path, labels = Path(path), []
    clean = lambda row: {k: ("" if v is None or v != v else v) for k, v in row.items()}
    if path.suffix.lower() == ".zip":
        import geopandas as gpd
        with zipfile.ZipFile(path) as z:
            for n in sorted(m for m in z.namelist() if m.lower().endswith(".shp")):
                recs = read_shp(z.read(n))
                attrs = gpd.read_file(f"zip://{path}!{n}", ignore_geometry=True)
                if len(attrs) != len(recs):
                    raise SystemExit(f"{n}: {len(recs)} shapes but {len(attrs)} attribute rows")
                labels += [{"props": clean(row), "kind": kind, "geom": geom}
                           for (kind, geom), row in zip(recs, attrs.to_dict("records")) if kind]
    else:
        for f in json.loads(path.read_text())["features"]:
            g, props = f.get("geometry") or {}, clean(f.get("properties") or {})
            if g.get("type") == "Point":
                labels.append({"props": props, "kind": "point", "geom": tuple(g["coordinates"][:2])})
            for poly in {"Polygon": [g.get("coordinates")], "MultiPolygon": g.get("coordinates")}.get(g.get("type"), []):
                labels.append({"props": dict(props), "kind": "polygon", "geom": [np.asarray(r, float)[:, :2] for r in poly]})
    for l in labels:
        l["props"]["name"] = str(l["props"].get("name") or l["props"].get("label_name") or "Imported").strip()
    return labels


def without_holes(poly):
    """Cut a polygon (pixel coordinates) at integer columns until no piece has a hole."""
    from shapely.geometry import box
    if len(poly.interiors) == 0:
        return [poly]
    x0, _, x1, _ = poly.interiors[0].bounds
    x = int(round(x0 + max(1, (x1 - x0) // 2)))
    minx, miny, maxx, maxy = poly.bounds
    out = []
    for half in (box(minx - 1, miny - 1, x, maxy + 1), box(x, miny - 1, maxx + 1, maxy + 1)):
        piece = poly.intersection(half)
        for p in getattr(piece, "geoms", [piece]):
            if p.geom_type == "Polygon" and p.area > 0:
                out += without_holes(p)
    return out


def rings_to_geometry(rings):
    """One record's rings as a shapely geometry, filled even-odd whatever their orientation. A single
    ring stays as drawn: buffer(0) would drop one lobe of a self-intersecting hand-drawn outline,
    while TEE (and rasterio) fill both."""
    from shapely.geometry import Polygon
    if len(rings) == 1:
        return Polygon(rings[0])
    region = Polygon()
    for r in rings:  # TEE writes outer rings counter-clockwise, shapefile readers expect clockwise
        region = region.symmetric_difference(Polygon(r).buffer(0))
    return region


def ring_mask(rings, gt, shape):
    """Even-odd fill of the rings of one record on the viewport grid (pixel-centre rule)."""
    from affine import Affine
    from rasterio.features import rasterize
    T = Affine(gt["a"], 0.0, gt["c"], 0.0, gt["e"], gt["f"])
    return rasterize([(rings_to_geometry(rings), 1)], out_shape=shape, transform=T, fill=0, dtype="uint8").astype(bool)


def mask_polygons(mask, gt):
    """A pixel mask as rings along the pixel edges (4-connected pieces, holes cut open)."""
    from rasterio.features import shapes
    from shapely.geometry import shape as to_shape
    rings = []
    for geom, v in shapes(mask.astype("uint8"), mask=mask, connectivity=4):
        if v == 1:
            for p in without_holes(to_shape(geom)):
                rings.append([[gt["c"] + x * gt["a"], gt["f"] + y * gt["e"]] for x, y in p.exterior.coords])
    return rings


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("export"); ap.add_argument("out")
    ap.add_argument("--metadata", default=str(Path.home() / "tee_data/vectors/Gteborg_GT/2024/metadata.json"))
    ap.add_argument("--types", default="")
    ap.add_argument("--regions-out", default="")
    a = ap.parse_args()
    keep = {t.strip() for t in a.types.split(",") if t.strip()}
    labels = read_export(a.export)
    typ_of = lambda l: l["props"].get("type") or ("point" if l["kind"] == "point" else "polygon")
    is_pixels = lambda l: l["kind"] == "polygon" and (typ_of(l) == "similarity" or len(l["geom"]) > 1)
    gt = shape = None
    if any(is_pixels(l) for l in labels):
        if not Path(a.metadata).exists():
            raise SystemExit(f"pixel labels need the viewport grid: {a.metadata} not found (--metadata)")
        meta = json.loads(Path(a.metadata).read_text())
        gt, shape = meta["geotransform"], (meta["mosaic_height"], meta["mosaic_width"])

    feats, regions, stats, masks = [], [], {}, {}
    for l in labels:
        p, typ = l["props"], typ_of(l)
        is_region = bool(TEAM.search(p["name"]))
        if keep and typ not in keep and not is_region:
            continue
        key = (p["name"], typ)
        stats.setdefault(key, [0, 0])[0] += 1
        if is_pixels(l) and not is_region:  # collected per class, drawn after the loop
            m = masks.setdefault(key, [p, np.zeros(shape, bool)])
            m[1] |= ring_mask(l["geom"], gt, shape)
            continue
        if l["kind"] == "point":
            g = {"type": "Point", "coordinates": [float(l["geom"][0]), float(l["geom"][1])]}
        else:
            g = {"type": "Polygon", "coordinates": [l["geom"][0].tolist()]}
        (regions if is_region else feats).append({"type": "Feature", "properties": p, "geometry": g})
        stats[key][1] += 1
    for key, (p, mask) in masks.items():
        rings = mask_polygons(mask, gt)
        feats += [{"type": "Feature", "properties": p, "geometry": {"type": "Polygon", "coordinates": [r]}} for r in rings]
        stats[key][1] += len(rings)
        print(f"  {key[0]} ({key[1]}): {int(mask.sum())} pixels on the {shape[1]} x {shape[0]} grid")

    for (name, typ), (n_in, n_out) in sorted(stats.items()):
        print(f"  {name:22s} {typ:10s} {n_in:5d} records -> {n_out:5d} features")
    Path(a.out).write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    print(f"wrote {a.out}: {len(feats)} features")
    if a.regions_out:
        Path(a.regions_out).write_text(json.dumps({"type": "FeatureCollection", "features": regions}))
        print(f"wrote {a.regions_out}: {len(regions)} region outline(s)")
    elif regions:
        print(f"left out {len(regions)} region outline(s): {sorted({r['properties']['name'] for r in regions})}")


if __name__ == "__main__":
    main()
