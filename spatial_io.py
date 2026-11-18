"""
Minimal GeoPackage reader (no geopandas needed): reads one feature layer with sqlite3 and
decodes the polygon geometries (Polygon / MultiPolygon, optionally with Z) from WKB.

A geometry is returned as a list of polygons; each polygon is a list of rings
(numpy arrays of shape (k, 2)); ring 0 is the exterior, the others are holes.
"""
import sqlite3
import struct

import numpy as np
import pandas as pd

_ENV = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}


def decode_gpkg_geometry(blob):
    b = bytes(blob)
    w = b[8 + _ENV[(b[3] >> 1) & 7]:]
    bo = "<" if w[0] == 1 else ">"
    t = struct.unpack(bo + "I", w[1:5])[0]
    dim = 3 if t >= 1000 else 2          # 1003/1006 = Z geometries
    t %= 1000

    def read_polygon(off):
        nr = struct.unpack(bo + "I", w[off:off + 4])[0]
        off += 4
        rings = []
        for _ in range(nr):
            n = struct.unpack(bo + "I", w[off:off + 4])[0]
            off += 4
            arr = np.frombuffer(w[off:off + 8 * dim * n], dtype=bo + "f8").reshape(-1, dim)
            rings.append(arr[:, :2].copy())
            off += 8 * dim * n
        return rings, off

    if t == 3:
        return [read_polygon(5)[0]]
    n = struct.unpack(bo + "I", w[5:9])[0]
    off, polys = 9, []
    for _ in range(n):
        off += 5                           # byte order + geometry type of the sub-polygon
        rings, off = read_polygon(off)
        polys.append(rings)
    return polys


def read_layer(path, table, columns):
    con = sqlite3.connect(str(path))
    df = pd.read_sql(f"select {', '.join(columns)}, geom from '{table}'", con)
    con.close()
    df["polys"] = [decode_gpkg_geometry(g) for g in df.pop("geom")]
    return df


def ring_area(r):
    x, y = r[:, 0], r[:, 1]
    return 0.5 * (np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))


def polys_area(polys):
    return sum(abs(ring_area(p[0])) - sum(abs(ring_area(h)) for h in p[1:]) for p in polys)


def polys_centroid(polys):
    """Area-weighted centroid of the exterior rings."""
    A = cx = cy = 0.0
    for p in polys:
        r = p[0]
        x, y = r[:, 0], r[:, 1]
        cr = x[:-1] * y[1:] - x[1:] * y[:-1]
        a = cr.sum() / 2
        if a == 0:
            continue
        A += a
        cx += ((x[:-1] + x[1:]) * cr).sum() / 6
        cy += ((y[:-1] + y[1:]) * cr).sum() / 6
    return (cx / A, cy / A) if A else (np.nan, np.nan)
