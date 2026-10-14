"""Draft Gothenburg labels from OpenStreetMap in Alice's class scheme, to be checked by hand in TEE
(Christopher, 12 Oct 2026).

    ~/tee-local/venv/bin/python osm_prelabel.py --alice EXPORT [EXPORT ...] --out-dir data
        [--osm ~/tee_data/harness-cache/osm_gothenburg_gt.json] [--purity 0.5] [--fine 5]
        [--metadata ~/tee_data/vectors/Gteborg_GT/2024/metadata.json]

OSM features (buildings, water, wood / forest, soccer pitches, roads buffered to their width) are
burnt on a grid `--fine` times finer than the viewport grid. A viewport pixel gets a class when that
class covers at least `--purity` of its area, so mixed pixels stay unlabelled. The draft is compared
with the hand-drawn polygons in --alice (TEE exports, ZIP or GeoJSON; one of them must contain the
region outline named after team members) and written per region as GeoJSON that TEE's
"Import -> From file..." reads; pixels that already have a hand-drawn label are left out.
The OSM extract is cached in --osm and fetched from the Overpass API when the file is missing.
"""
import argparse, json, re, sys, warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from tee_export_to_geojson import TEAM, mask_polygons, read_export, ring_mask  # noqa: E402

warnings.filterwarnings("ignore")
# class -> colour in Alice's export; the order is the burn order (later classes win in a fine pixel)
CLASSES = {"FOREST": "#8a3dd6", "RIVER": "#1036f4", "FOOTBALL FIELD": "#e6194b", "STREET": "#b771aa", "BUILDINGS": "#3cb44b"}
# carriageway width in metres when a road has neither width=* nor lanes=* (one OSM way per direction on motorways)
ROAD_WIDTH = {"motorway": 11, "trunk": 9, "primary": 9, "secondary": 8, "tertiary": 7, "residential": 6, "unclassified": 6,
              "living_street": 6, "pedestrian": 5, "service": 4, "busway": 4, "motorway_link": 5, "trunk_link": 5,
              "primary_link": 5, "secondary_link": 5, "tertiary_link": 5}
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter"]
QUERY = """[out:json][timeout:120][bbox:{s},{w},{n},{e}];
(way["building"]; relation["building"]["type"="multipolygon"];
 way["natural"="water"]; relation["natural"="water"];
 way["waterway"~"^(riverbank|dock)$"]; relation["waterway"~"^(riverbank|dock)$"];
 way["natural"="wood"]; relation["natural"="wood"]; way["landuse"="forest"]; relation["landuse"="forest"];
 way["leisure"="pitch"]; relation["leisure"="pitch"]; way["highway"];);
out geom;"""


def fetch(path, bounds):
    import requests
    w, s, e, n = bounds
    for url in OVERPASS:
        r = requests.post(url, data={"data": QUERY.format(s=s, w=w, n=n, e=e)}, timeout=200,
                          headers={"User-Agent": "TESSERA clusterings student project (Chalmers DAT300)"})
        if r.status_code == 200 and r.text.lstrip().startswith("{"):
            Path(path).write_text(r.text)
            return
        print(f"  {url}: HTTP {r.status_code}, trying the next server")
    raise SystemExit("no Overpass server answered; try again later")


def osm_class(tags):
    if "building" in tags:
        return "BUILDINGS"
    if tags.get("natural") == "water" or tags.get("waterway") in ("riverbank", "dock"):
        return "RIVER"
    if tags.get("natural") == "wood" or tags.get("landuse") == "forest":
        return "FOREST"
    if tags.get("leisure") == "pitch" and "soccer" in tags.get("sport", ""):
        return "FOOTBALL FIELD"
    return None


def road_width(tags):
    """Width in metres of a road that cars use above ground, else None."""
    if tags.get("highway") not in ROAD_WIDTH or tags.get("tunnel") or tags.get("covered") == "yes":
        return None
    m = re.match(r"\s*(\d+(\.\d+)?)", tags.get("width", ""))
    if m and 2 <= float(m.group(1)) <= 40:
        return float(m.group(1))
    m = re.match(r"\s*(\d+)", tags.get("lanes", ""))
    return 3.5 * int(m.group(1)) + 1 if m else ROAD_WIDTH[tags["highway"]]


def osm_geometries(elements):
    """{class: [shapely geometry in EPSG:4326]}; roads are buffered to their width in SWEREF 99 TM."""
    import geopandas as gpd
    from shapely.geometry import LineString, Polygon
    from shapely.ops import polygonize, unary_union
    line = lambda geom: [(p["lon"], p["lat"]) for p in geom]
    out, roads = {c: [] for c in CLASSES}, []
    for e in elements:
        tags = e.get("tags", {})
        cls = osm_class(tags)
        if e["type"] == "way":
            pts = line(e["geometry"])
            closed = len(pts) >= 4 and pts[0] == pts[-1]
            if cls and closed:
                out[cls].append(Polygon(pts))
            elif "highway" in tags and road_width(tags):
                if closed and tags.get("area") == "yes":
                    out["STREET"].append(Polygon(pts))
                elif len(pts) >= 2:
                    roads.append((LineString(pts), road_width(tags)))
        elif cls:  # multipolygon relation: outer rings minus inner rings
            parts = {"outer": [], "inner": []}
            for m in e.get("members", []):
                if m["type"] == "way" and m.get("geometry") and m.get("role") in parts:
                    parts[m["role"]].append(LineString(line(m["geometry"])))
            outer = unary_union(list(polygonize(unary_union(parts["outer"])))) if parts["outer"] else None
            if outer is not None and not outer.is_empty:
                inner = unary_union(list(polygonize(unary_union(parts["inner"])))) if parts["inner"] else None
                out[cls].append(outer.difference(inner) if inner is not None else outer)
    g = gpd.GeoDataFrame({"w": [w for _, w in roads]}, geometry=[l for l, _ in roads], crs=4326).to_crs(3006)
    out["STREET"] += list(g.geometry.buffer(g["w"] / 2, cap_style="flat").to_crs(4326))
    return {c: [x.buffer(0) for x in v if not x.is_empty] for c, v in out.items()}


def class_fractions(geoms, gt, shape, fine):
    """Share of every viewport pixel covered by each class: array [class, row, col]."""
    from affine import Affine
    from rasterio.features import rasterize
    T = Affine(gt["a"] / fine, 0.0, gt["c"], 0.0, gt["e"] / fine, gt["f"])
    hi = (shape[0] * fine, shape[1] * fine)
    burn = [(g, i) for i, c in enumerate(CLASSES, 1) for g in geoms[c]]
    r = rasterize(burn, out_shape=hi, transform=T, fill=0, dtype="uint8")
    blocks = r.reshape(shape[0], fine, shape[1], fine)
    return np.stack([(blocks == i).mean(axis=(1, 3)) for i in range(1, len(CLASSES) + 1)])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--alice", required=True, nargs="+")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--osm", default=str(Path.home() / "tee_data/harness-cache/osm_gothenburg_gt.json"))
    ap.add_argument("--metadata", default=str(Path.home() / "tee_data/vectors/Gteborg_GT/2024/metadata.json"))
    ap.add_argument("--purity", type=float, default=0.5)
    ap.add_argument("--fine", type=int, default=5)
    a = ap.parse_args()
    meta = json.loads(Path(a.metadata).read_text())
    gt, shape = meta["geotransform"], (meta["mosaic_height"], meta["mosaic_width"])
    names = list(CLASSES)
    if not Path(a.osm).exists():
        fetch(a.osm, (gt["c"], gt["f"] + shape[0] * gt["e"], gt["c"] + shape[1] * gt["a"], gt["f"]))
    osm = json.loads(Path(a.osm).read_text())
    print(f"OSM extract of {osm.get('osm3s', {}).get('timestamp_osm_base', '?')}: {len(osm['elements'])} elements")
    geoms = osm_geometries(osm["elements"])
    frac = class_fractions(geoms, gt, shape, a.fine)
    draft = np.where(frac.max(0) >= a.purity, frac.argmax(0) + 1, 0)  # 0 = no class, else index into names + 1
    print(f"draft at purity >= {a.purity}: {100 * (draft > 0).mean():.1f} % of the viewport")

    # Alice's export: hand-drawn polygons per class, cluster pixels, region outline
    hand = np.zeros((len(names),) + shape, bool)
    cluster, region = np.zeros(shape, bool), np.zeros(shape, bool)
    for l in [l for f in a.alice for l in read_export(f)]:
        n = l["props"]["name"].upper()
        if l["kind"] != "polygon":
            continue
        m = ring_mask(l["geom"], gt, shape)
        if TEAM.search(n):
            region |= m
        elif l["props"].get("type") == "polygon" and n in names:
            hand[names.index(n)] |= m
        else:
            cluster |= m
    if not region.any():
        raise SystemExit("no region outline (a feature named after team members) in --alice")
    one = hand.sum(0) == 1  # pixels that Alice gave exactly one class by hand
    alice = np.where(one, hand.argmax(0) + 1, 0)

    print("\nAlice's hand-drawn pixels (rows) against the OSM draft (columns), in pixels:")
    print(f"{'':16s}" + "".join(f"{n[:9]:>10s}" for n in names) + f"{'no class':>10s}{'same':>8s}{'other':>8s}")
    both = same = 0
    for i, n in enumerate(names, 1):
        row = [int(((alice == i) & (draft == j)).sum()) for j in range(1, len(names) + 1)]
        none, tot = int(((alice == i) & (draft == 0)).sum()), int((alice == i).sum())
        both += sum(row); same += row[i - 1]
        print(f"{n:16s}" + "".join(f"{v:10d}" for v in row) + f"{none:10d}{100 * row[i - 1] / max(tot, 1):7.1f}%{100 * (sum(row) - row[i - 1]) / max(tot, 1):7.1f}%")
    print(f"labelled by both: {both} px, same class on {same} = {100 * same / max(both, 1):.1f} %")
    print("share of the draft that agrees where Alice has a hand label (per draft class):")
    for j, n in enumerate(names, 1):
        d = (draft == j) & (alice > 0)
        print(f"    {n:16s} {100 * (alice[d] == j).mean() if d.any() else float('nan'):5.1f} % of {int(d.sum())} px")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for tag, area in (("north", region), ("south", ~region)):
        feats, free = [], (draft > 0) & area & ~hand.any(0)
        for j, n in enumerate(names, 1):
            m = free & (draft == j)
            props = {"name": n, "color": CLASSES[n], "code": "", "type": "osm"}
            feats += [{"type": "Feature", "properties": props, "geometry": {"type": "Polygon", "coordinates": [r]}} for r in mask_polygons(m, gt)]
        f = out / f"osm_draft_{tag}.geojson"
        f.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
        per = ", ".join(f"{n} {int((free & (draft == j)).sum())}" for j, n in enumerate(names, 1))
        print(f"\n{tag}: {int(area.sum())} px; by hand {100 * hand.any(0)[area].mean():.1f} %, cluster only {100 * (cluster & ~hand.any(0))[area].mean():.1f} %, "
              f"new from OSM {100 * free[area].mean():.1f} % ({per})\n    wrote {f} ({len(feats)} polygons)")
    np.savez_compressed(out / "osm_draft.npz", frac=frac.astype("float32"), draft=draft.astype("uint8"), hand=hand,
                        cluster=cluster, region=region, classes=np.array(names))


if __name__ == "__main__":
    main()
