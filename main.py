import json, os, random, tempfile, threading, time, webbrowser
from dataclasses import dataclass, asdict
from typing import Dict
import cv2
import numpy as np
from PIL import Image, ImageDraw
from perlin_noise import PerlinNoise
import gradio as gr

APP_TITLE = "MAGER - Procedural Map Studio"
GRADIO_PORT = 7869
HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(HERE, "settings.json")
DEFAULT_EXPORT_DIR = os.path.join(HERE, "exports")

def _rel(p):
    """Display path relative to the app folder so the user's local
    path (C:\\Users\\...) is never shown. Falls back to the input
    when the path lives outside the app folder."""
    try:
        rel = os.path.relpath(os.path.abspath(p), HERE)
        if not rel.startswith(".."):
            return rel
    except Exception:
        pass
    return p

def _load_settings():
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_settings(s):
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2)
    except Exception:
        pass

def get_export_dir():
    d = (_load_settings().get("export_dir") or "").strip() or DEFAULT_EXPORT_DIR
    if not os.path.isabs(d):
        d = os.path.join(HERE, d)  # relative value in settings.json
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        d = DEFAULT_EXPORT_DIR
        os.makedirs(d, exist_ok=True)
    return d

def set_export_dir(path):
    path = (path or "").strip()
    if not path:
        raise gr.Error("Pick a folder first (Browse...).")
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        raise gr.Error("Cannot use folder %s: %s" % (path, e))
    s = _load_settings()
    # store relative when inside the app folder (no local path on disk)
    try:
        rel = os.path.relpath(os.path.abspath(path), HERE)
        s["export_dir"] = rel if not rel.startswith("..") else os.path.abspath(path)
    except Exception:
        s["export_dir"] = os.path.abspath(path)
    _save_settings(s)
    return "Export folder:\n" + _rel(path)
PAL = [(41,171,226),(67,175,105),(237,185,27),(155,89,182),(231,76,60),(52,73,94),(26,188,156),(243,156,18),(211,84,0),(127,140,141),(44,62,80),(192,57,43)]
NAMES = ["Water","Forest","Sand","Mountain","Grass","Rock","Snow","Swamp","Road","Urban","Farm","Lava"]
SCAN = {}
GEN = {"entries": None, "grid": None, "seed": None}

def bgr_hex(b):
    bb, gg, rr = (int(x) for x in b)
    return "#%02X%02X%02X" % (rr, gg, bb)

def hex_rgb(h):
    h = h.lstrip("#")
    return (int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))

@dataclass
class Res:
    id: int
    label: str
    color_hex: str
    mean_color_hex: str
    coverage: float
    threshold: float

def to_json(es):
    return {"meta": {"app": "MAGER", "version": 1},
            "resources": [asdict(e) for e in es]}

def from_json(data):
    out = []
    for r in data.get("resources") or data.get("materials") or []:
        out.append(Res(int(r["id"]), str(r.get("label","seg")),
            str(r.get("color_hex","#808080")),
            str(r.get("mean_color_hex","#808080")),
            float(r.get("coverage",0)), float(r.get("threshold",0))))
    return out


def _read_json_payload(jf):
    """Return (es, payload) from a JSON input.

    es is a list of Res (empty if the JSON holds no readable resources).
    payload is the raw parsed dict, which may carry a 'texture' field that
    Feature 2 can render from instead of only deriving colours from materials.
    """
    try:
        if jf is None or (isinstance(jf, str) and not jf.strip()):
            return [], None
        if hasattr(jf, "read"):  # file-like object
            data = json.load(jf)
        else:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
    except Exception:
        return [], None
    es = from_json(data)
    return es, data

def def_th(n):
    if n <= 1:
        return [0.0]
    return [round(1.0 - i*(1.0/n), 3) for i in range(n)]

def seg_km(img, k, morph=5):
    h, w, _ = img.shape
    k = max(2, min(int(k), 12))
    work = cv2.bilateralFilter(img, 7, 60, 60)
    sc = min(1.0, 320.0/max(h, w))
    small = cv2.resize(work, (int(w*sc), int(h*sc)), interpolation=cv2.INTER_AREA)
    px = small.reshape(-1,3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, _, cent = cv2.kmeans(px, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    cent = cent.astype(np.uint8)
    flat = work.reshape(-1,3).astype(np.int16)
    c = cent.astype(np.int16)
    lab = np.empty(flat.shape[0], dtype=np.int32)
    for i in range(0, flat.shape[0], 200000):
        dd = np.linalg.norm(flat[i:i+200000,None,:]-c[None,:,:], axis=2)
        lab[i:i+200000] = np.argmin(dd, axis=1)
    lm = lab.reshape(h, w)
    m = max(0, int(morph))
    if m >= 3:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (m, m))
        for i in range(k):
            msk = (lm==i).astype(np.uint8)*255
            msk = cv2.morphologyEx(msk, cv2.MORPH_OPEN, ker)
            msk = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, ker)
            keep = (msk>0)
            lm[(lm==i) & (~keep)] = -1
        holes = (lm==-1)
        if holes.any():
            fh = work[holes].astype(np.int16)
            cc = cent.astype(np.int16)
            dd = np.linalg.norm(fh[:,None,:]-cc[None,:,:], axis=2)
            lm[holes] = np.argmin(dd, axis=1)
    vis = np.zeros_like(img)
    for i in range(k):
        b, g, r = PAL[i % len(PAL)]
        vis[lm==i] = (r, g, b)
    ed = cv2.Canny((lm.astype(np.uint8)*(255//max(k-1,1))), 50, 150)
    vis[ed>0] = (255,255,255)
    cnt = np.bincount(lm.ravel(), minlength=k)
    tot = cnt.sum()
    info = []
    for i in range(k):
        mr, mg, mb = (int(x) for x in cent[i])
        info.append({"id": i, "mean": "#%02X%02X%02X" % (mr, mg, mb),
                     "cov": float(cnt[i]/tot), "kind": "lump",
                     "area": int(cnt[i]), "cx": -1, "cy": -1})
    return vis, info
def _lab_means(work, lm, k):
    lab = cv2.cvtColor(work, cv2.COLOR_RGB2LAB).astype(np.float32)
    ms = []
    for i in range(k):
        m = (lm == i)
        ms.append(lab[m].mean(axis=0) if m.any() else np.zeros(3, np.float32))
    return np.stack(ms)

def _lump_tex(gray, mask):
    m = mask > 0
    if not m.any():
        return np.array([0.0, 0.0], np.float64)
    px = gray[m].astype(np.float64)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)[m].mean() / 255.0
    return np.array([px.std() / 255.0, edge], np.float64)

def seg_patterns(img, k, morph=7, min_area=400, merge=1.0):
    h, w, _ = img.shape
    k = max(2, min(int(k), 12))
    work = cv2.bilateralFilter(img, 7, 60, 60)
    gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
    sc = min(1.0, 320.0 / max(h, w))
    small = cv2.resize(work, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    px = small.reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, _, cent = cv2.kmeans(px, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    cent = cent.astype(np.uint8)
    flat = work.reshape(-1, 3).astype(np.int16)
    c = cent.astype(np.int16)
    lab = np.empty(flat.shape[0], dtype=np.int32)
    for i in range(0, flat.shape[0], 200000):
        dd = np.linalg.norm(flat[i:i + 200000, None, :] - c[None, :, :], axis=2)
        lab[i:i + 200000] = np.argmin(dd, axis=1)
    lm = lab.reshape(h, w)
    m = max(3, int(morph))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (m, m))
    blobs = []
    for ci in range(k):
        msk = (lm == ci).astype(np.uint8) * 255
        msk = cv2.morphologyEx(msk, cv2.MORPH_OPEN, ker)
        msk = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, ker)
        n, cc, stats, c2 = cv2.connectedComponentsWithStats(msk, 8)
        for j in range(1, n):
            area = int(stats[j, cv2.CC_STAT_AREA])
            if area < int(min_area):
                continue
            blobs.append((ci, (cc == j), area, int(c2[j][0]), int(c2[j][1])))
    if not blobs:
        return seg_km(img, k, morph=morph)
    lmeans = _lab_means(work, lm, k)
    feats = np.stack([_lump_tex(gray, b[1].astype(np.uint8) * 255) for b in blobs])
    tcol = 28.0 / max(0.3, float(merge))
    ttex = 0.35 / max(0.3, float(merge))
    parent = list(range(len(blobs)))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a in range(len(blobs)):
        for b in range(a + 1, len(blobs)):
            dc = float(np.linalg.norm(lmeans[blobs[a][0]] - lmeans[blobs[b][0]]))
            dt = float(np.linalg.norm(feats[a] - feats[b]))
            if dc <= tcol and dt <= ttex:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
    groups = {}
    for i in range(len(blobs)):
        groups.setdefault(find(i), []).append(i)
    ordered = sorted(groups.values(), key=lambda g: sum(blobs[i][2] for i in g), reverse=True)
    return _paint_groups(img, blobs, cent, ordered, ker)
def _paint_groups(img, blobs, cent, ordered, ker):
    h, w, _ = img.shape
    vis = np.zeros_like(img)
    gid_map = np.full((h, w), -1, dtype=np.int32)
    info = []
    for gid, g in enumerate(ordered):
        b, gg, r = PAL[gid % len(PAL)]
        tot = 0
        sx, sy = 0, 0
        acc = np.zeros(3, np.float64)
        for i in g:
            ci, mask, area, cx, cy = blobs[i]
            gid_map[mask] = gid
            tot += area
            sx += cx * area
            sy += cy * area
            acc += cent[ci].astype(np.float64) * area
        mc = (acc / max(tot, 1)).astype(int)
        vis[gid_map == gid] = (r, gg, b)
        info.append({"id": gid, "mean": "#%02X%02X%02X" % (int(mc[0]), int(mc[1]), int(mc[2])),
                     "cov": float(tot / (h * w)), "kind": "pattern",
                     "area": int(tot), "cx": int(sx / max(tot, 1)),
                     "cy": int(sy / max(tot, 1)), "parts": len(g)})
    if (gid_map == -1).any():
        filled = gid_map.copy()
        masks = [(filled == o).astype(np.uint8) * 255 for o in range(len(ordered))]
        while (filled == -1).any():
            grown = False
            for o in range(len(ordered)):
                dil = cv2.dilate(masks[o], ker)
                new = (dil > 0) & (filled == -1)
                if new.any():
                    filled[new] = o
                    masks[o] = (filled == o).astype(np.uint8) * 255
                    grown = True
            if not grown:
                filled[filled == -1] = 0
                break
        gid_map = filled
        for o in range(len(ordered)):
            b, gg, r = PAL[o % len(PAL)]
            vis[gid_map == o] = (r, gg, b)
    for o in range(len(ordered)):
        m2 = (gid_map == o).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (255, 255, 255), 2)
    for s in info:
        cv2.putText(vis, str(s["id"]), (max(0, s["cx"] - 8), max(10, s["cy"] + 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, str(s["id"]), (max(0, s["cx"] - 8), max(10, s["cy"] + 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cnt = np.bincount(gid_map.ravel(), minlength=len(ordered))
    tot = max(cnt.sum(), 1)
    for s in info:
        s["cov"] = float(cnt[s["id"]] / tot)
    SCAN["obj_map"] = gid_map
    return vis, info
def seg_nowrap_marker():  # placeholder replaced below
    return None
def seg_objects(img, k, morph=7, min_area=400):
    h, w, _ = img.shape
    k = max(2, min(int(k), 12))
    work = cv2.bilateralFilter(img, 7, 60, 60)
    sc = min(1.0, 320.0/max(h, w))
    small = cv2.resize(work, (int(w*sc), int(h*sc)), interpolation=cv2.INTER_AREA)
    px = small.reshape(-1,3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, _, cent = cv2.kmeans(px, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    cent = cent.astype(np.uint8)
    flat = work.reshape(-1,3).astype(np.int16)
    c = cent.astype(np.int16)
    lab = np.empty(flat.shape[0], dtype=np.int32)
    for i in range(0, flat.shape[0], 200000):
        dd = np.linalg.norm(flat[i:i+200000,None,:]-c[None,:,:], axis=2)
        lab[i:i+200000] = np.argmin(dd, axis=1)
    lm = lab.reshape(h, w)
    m = max(3, int(morph))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (m, m))
    obj_id = np.full((h, w), -1, dtype=np.int32)
    info = []
    vis = np.zeros_like(img)
    oid = 0
    for ci in range(k):
        msk = (lm==ci).astype(np.uint8)*255
        msk = cv2.morphologyEx(msk, cv2.MORPH_OPEN, ker)
        msk = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, ker)
        n, cc, stats, c2 = cv2.connectedComponentsWithStats(msk, 8)
        mr, mg, mb = (int(x) for x in cent[ci])
        for j in range(1, n):
            area = int(stats[j, cv2.CC_STAT_AREA])
            if area < int(min_area):
                continue
            comp = (cc==j)
            obj_id[comp] = oid
            b, g, r = PAL[oid % len(PAL)]
            vis[comp] = (r, g, b)
            cx, cy = int(c2[j][0]), int(c2[j][1])
            x = int(stats[j, cv2.CC_STAT_LEFT])
            y = int(stats[j, cv2.CC_STAT_TOP])
            ww = int(stats[j, cv2.CC_STAT_WIDTH])
            hh = int(stats[j, cv2.CC_STAT_HEIGHT])
            info.append({"id": oid, "mean": "#%02X%02X%02X" % (mr, mg, mb),
                         "cov": float(area/(h*w)), "kind": "object",
                         "area": area, "cx": cx, "cy": cy,
                         "bbox": [x, y, ww, hh], "color_idx": ci})
            oid += 1
    if oid == 0:
        return seg_km(img, k, morph=morph)
    un = (obj_id==-1)
    if un.any():
        # fill speckle gaps: repeatedly dilate each object mask, first-writer-wins
        filled = obj_id.copy()
        masks = [(filled==o).astype(np.uint8)*255 for o in range(oid)]
        while (filled==-1).any():
            grown = False
            for o in range(oid):
                dil = cv2.dilate(masks[o], ker)
                new = (dil>0) & (filled==-1)
                if new.any():
                    filled[new] = o
                    masks[o] = (filled==o).astype(np.uint8)*255
                    grown = True
            if not grown:
                filled[filled==-1] = 0
                break
        obj_id = filled
        for o in range(oid):
            b, g, r = PAL[o % len(PAL)]
            vis[obj_id==o] = (r, g, b)
    for o in range(oid):
        m2 = (obj_id==o).astype(np.uint8)*255
        cnts, _ = cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (255,255,255), 2)
    for s in info:
        cv2.putText(vis, str(s["id"]), (max(0,s["cx"]-8), max(10,s["cy"]+6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(vis, str(s["id"]), (max(0,s["cx"]-8), max(10,s["cy"]+6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    cnt = np.bincount(obj_id.ravel(), minlength=oid)
    tot = max(cnt.sum(), 1)
    for s in info:
        s["cov"] = float(cnt[s["id"]]/tot)
    info.sort(key=lambda s: s["area"], reverse=True)
    remap = {s["id"]: i for i, s in enumerate(info)}
    nid = np.full_like(obj_id, -1)
    for old, new in remap.items():
        nid[obj_id==old] = new
    for i, s in enumerate(info):
        s["id"] = i
    SCAN["obj_map"] = nid
    return vis, info

def norm_img(image):
    if image is None:
        return None
    a = np.asarray(image)
    if a.size == 0:
        return None
    if a.dtype != np.uint8:
        if a.max() <= 1.0:
            a = (a*255.0).clip(0, 255)
        a = a.astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    elif a.ndim == 3 and a.shape[2] == 4:
        _rgb = a[:, :, :3].astype(np.int16)
        _al = a[:, :, 3:4].astype(np.int16)
        a = ((_rgb * _al + 255 * (255 - _al)) // 255).astype(np.uint8)
    elif a.ndim == 3 and a.shape[2] > 4:
        a = a[:, :, :3]
    if a.ndim != 3 or a.shape[2] != 3:
        raise gr.Error("Unsupported image shape %s. Paste/upload a normal RGB image." % (a.shape,))
    return np.ascontiguousarray(a)

def blend_overlay(base, orig, alpha):
    try:
        a = max(0.0, min(1.0, float(alpha)))
    except Exception:
        a = 0.0
    if base is None:
        return None
    if orig is None or a <= 0.0:
        return base
    if orig.shape[:2] != base.shape[:2]:
        orig = cv2.resize(orig, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_AREA)
    return ((base.astype(np.float32)*(1.0-a) + orig.astype(np.float32)*a)).astype(np.uint8)

def f1_overlay(alpha):
    if SCAN.get("vis") is None:
        raise gr.Error("Run Scan first.")
    return blend_overlay(SCAN["vis"], SCAN.get("orig"), alpha)

def _to_b64(img):
    import base64, io
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def _empty_view():
    return ("<div style='color:#888;padding:20px;text-align:center'>"
            "Run <b>Scan Image</b> - tint + real map appear here, slider blends instantly.</div>")

def _spotlight(vis, orig, gid, alpha=0.35):
    if vis is None or orig is None:
        return None
    m = (SCAN.get("obj_map") == gid) if SCAN.get("obj_map") is not None else None
    try:
        a = max(0.0, min(1.0, float(alpha)))
    except Exception:
        a = 0.35
    base = blend_overlay(vis, orig, a)
    if m is None or not m.any():
        return base
    dark = (base.astype(np.float32) * 0.25).astype(np.uint8)
    out = dark
    out[m] = base[m]
    cnts, _ = cv2.findContours(m.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255, 255, 0), 2)
    return out

def _thumbs(vis, orig, info, size=160):
    if vis is None or orig is None or not info:
        return []
    h, w, _ = vis.shape
    sc = min(1.0, size / max(h, w))
    nw, nh = max(1, int(w * sc)), max(1, int(h * sc))
    om = SCAN.get("obj_map")
    thumbs = []
    for s in info:
        gid = s["id"]
        m = (om == gid) if om is not None else None
        combo = np.hstack([cv2.resize(orig, (nw, nh), interpolation=cv2.INTER_AREA),
                           cv2.resize(vis, (nw, nh), interpolation=cv2.INTER_NEAREST)])
        if m is not None and m.any():
            mm = cv2.resize(m.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0
            left = combo[:, :nw].copy()
            dark = (left.astype(np.float32) * 0.25).astype(np.uint8)
            dark[mm] = left[mm]
            cnts, _ = cv2.findContours(mm.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(dark, cnts, -1, (255, 255, 0), 1)
            combo[:, :nw] = dark
        thumbs.append((combo, "ID %d | real (spotlight) | tint" % gid))
    return thumbs

def _locate(gid_str, alpha=0.35):
    if SCAN.get("vis") is None:
        raise gr.Error("Run Scan first.")
    try:
        gid = int(float(str(gid_str)))
    except Exception:
        raise gr.Error("Pick a segment ID.")
    return _spotlight(SCAN["vis"], SCAN.get("orig"), gid, alpha)

def _swatch_popup(gid, hexcol, vis, orig, size=128):
    om = SCAN.get("obj_map")
    if om is not None:
        m = (om == gid)
        combo = np.hstack([orig, vis]) if orig.shape == vis.shape else vis
        if m.any():
            ys, xs = np.where(m)
            y0, y1 = max(0, ys.min() - 4), min(combo.shape[0], ys.max() + 5)
            x0, x1 = max(0, xs.min() - 4), min(orig.shape[1], xs.max() + 5)
            crop = orig[y0:y1, x0:x1]
            dark = (crop.astype(np.float32) * 0.3).astype(np.uint8)
            mm = m[y0:y1, x0:x1]
            dark[mm] = crop[mm]
            cnts, _ = cv2.findContours(mm.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(dark, cnts, -1, (255, 255, 0), 1)
            thumb = cv2.resize(dark, (size, size), interpolation=cv2.INTER_NEAREST)
        else:
            thumb = cv2.resize(orig, (size, size), interpolation=cv2.INTER_AREA)
    else:
        thumb = cv2.resize(orig if orig is not None else vis, (size, size), interpolation=cv2.INTER_AREA)
    b64 = _to_b64(thumb)
    return (
        "<div class='swwrap' data-gid='" + str(gid) + "' data-thumb='" + b64 + "'"
        " style='position:relative;display:inline-block'>"
        "<div title='ID " + str(gid) + " - hover row for minimap' style='width:22px;height:22px;border-radius:4px;"
        "border:1px solid #555;background:" + hexcol + ";cursor:zoom-in'></div>"
        "</div>")

def _blend_view(tint_b64, real_b64, alpha):
    try:
        a = max(0.0, min(1.0, float(alpha)))
    except Exception:
        a = 0.35
    return (
        "<div style='position:relative;width:100%;background:#111'>"
        "<img src='" + tint_b64 + "' style='display:block;width:100%;image-rendering:pixelated'/>" +
        "<img id='reallay' src='" + real_b64 + "' style='position:absolute;inset:0;width:100%;height:100%;object-fit:fill;pointer-events:none;opacity:" + str(a) + "'/>" +
        "<script>(function(){var s=document.querySelector('#ovslider input[type=range]');"
        "var lay=document.getElementById('reallay');"
        "if(s&&lay){var f=function(){lay.style.opacity=(parseFloat(s.value)||0).toString();};"
        "s.addEventListener('input',f);f();}})();</script></div>")

def f1_scan(image, k, mode, morph, min_area, alpha, merge):
    image = norm_img(image)
    if image is None:
        return _empty_view(), [], \
            "Upload or paste an image first (upload / drag-drop / Ctrl+V)."
    ml = str(mode).lower()
    if ml.startswith("pattern"):
        vis, info = seg_patterns(image, int(k), morph=int(morph), min_area=int(min_area), merge=float(merge))
    elif ml.startswith("object"):
        vis, info = seg_objects(image, int(k), morph=int(morph), min_area=int(min_area))
    else:
        vis, info = seg_km(image, int(k), morph=int(morph))
    th = def_th(len(info))
    rows = []
    for i, s in enumerate(info):
        pb = PAL[s["id"] % len(PAL)]
        sw = _swatch_popup(s["id"], bgr_hex((int(pb[0]), int(pb[1]), int(pb[2]))), vis, image)
        rows.append([s["id"], s["mean"], round(s["cov"] * 100, 2),
                     NAMES[i % len(NAMES)], bgr_hex((int(pb[0]), int(pb[1]), int(pb[2]))), th[i], sw])
    SCAN["info"] = info
    SCAN["vis"] = vis
    SCAN["orig"] = image
    view = _blend_view(_to_b64(vis), _to_b64(image), alpha)
    return view, rows, "Found %d segments (%s). Hover a swatch for its minimap popup." % (len(info), mode)

def _save_export(name, write):
    p = os.path.join(get_export_dir(), name)
    write(p)
    return p

def _export_auto(name, write_file, kind):
    """Save straight to the configured export folder. No dialogs."""
    import datetime
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(name)
    p = _save_export("%s_%s%s" % (base, stamp, ext), write_file)
    return "OK - %s\nSaved: %s" % (kind, _rel(p))

def _ask_save_path(start_name):
    """Show native Save-As first, return chosen path or ''.

    Runs in a worker thread with a timeout so a blocked/cancelled
    dialog can never hang the Gradio server thread."""
    import concurrent.futures

    def _ask():
        try:
            import webview
            wins = getattr(webview, "windows", None) or []
            if not wins:
                return ""
            res = wins[0].create_file_dialog(webview.SAVE_DIALOG,
                                             save_filename=start_name)
            if not res:
                return ""
            return res[0] if isinstance(res, (list, tuple)) else res
        except Exception:
            return ""

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(_ask).result(timeout=120) or ""
    except Exception:
        return ""

def _export_then_save(start_name, write_file, kind):
    """Chooser FIRST, then write only once.

    - User picks destination -> bytes go straight there (single write).
    - Dialog cancelled/unavailable -> fall back to .\\exports\\ backup copy."""
    dest = _ask_save_path(start_name)
    if dest:
        try:
            write_file(dest)
            return "OK - %s\nSaved: %s" % (kind, dest)
        except Exception as e:
            raise gr.Error("Could not save to %s: %s" % (dest, e))
    import datetime
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(start_name)
    p = _save_export("%s_%s%s" % (base, stamp, ext), write_file)
    return ("OK - %s\nSaved (dialog cancelled/unavailable): %s\n"
            "(File is in .\\exports\\, use Reveal button.)" % (kind, p))

def _webview_save(src, start_name):
    """Ask pywebview for a native Save-As path and copy src there.

    Runs the dialog in a STA thread with a timeout so a blocked/cancelled
    dialog can never hang the Gradio server thread. Returns dest or ''."""
    import shutil
    import concurrent.futures

    def _ask():
        try:
            import webview
            wins = getattr(webview, "windows", None) or []
            if not wins:
                return ""
            res = wins[0].create_file_dialog(webview.SAVE_DIALOG,
                                             save_filename=start_name)
            if not res:
                return ""
            return res[0] if isinstance(res, (list, tuple)) else res
        except Exception:
            return ""

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            dest = ex.submit(_ask).result(timeout=120)
    except Exception:
        return ""
    if not dest:
        return ""
    try:
        shutil.copyfile(src, dest)
        return dest
    except Exception:
        return ""

def _export_done_msg(src, start_name, kind):
    dest = _webview_save(src, start_name)
    if dest:
        return "OK - %s\nSaved: %s" % (kind, dest)
    return ("OK - %s\nSaved: %s\n(Save-As dialog unavailable - file is in .\\exports\\, "
            "use Reveal button.)" % (kind, src))

def f1_export(table):
    # Gradio Dataframe arrives as pandas DataFrame -> convert to rows
    if table is not None and hasattr(table, "values"):
        try:
            table = table.values.tolist()
        except Exception:
            table = list(table)
    if table is None or len(table) == 0:
        raise gr.Error("Run Scan first.")
    es = []
    for row in table:
        try:
            vals = list(row)
        except Exception:
            continue
        if len(vals) < 7:
            continue
        # skip header row if present ("Segment ID", ...)
        if str(vals[0]).strip().lower() in ("segment id", "segment_id", "id", "s"):
            continue
        try:
            sid = int(float(str(vals[0]).strip()))
        except Exception:
            raise gr.Error("Bad Segment ID %r - must be a number." % (vals[0],))
        mh = str(vals[1])
        cov = float(vals[2])
        lab = str(vals[3]).strip() or ("seg_%d" % sid)
        chx = str(vals[4]).strip() or mh
        thr = max(0.0, min(1.0, float(vals[5])))
        es.append(Res(sid, lab, chx, mh, round(cov/100.0,5), thr))
    if not es:
        raise gr.Error("No valid rows. Run Scan first.")
    pay = to_json(es)
    def _write(p):
        open(p, "w", encoding="utf-8").write(json.dumps(pay, indent=2))
    try:
        hp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resource_label.json")
        open(hp,"w",encoding="utf-8").write(json.dumps(pay, indent=2))
    except Exception:
        pass
    return _export_auto("resource_label.json", _write, "resource_label.json exported.")
def load_entries(fobj):
    dflt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resource_label.json")
    p = fobj if fobj else (dflt if os.path.exists(dflt) else None)
    if not p or not os.path.exists(p):
        raise gr.Error("No resource_label.json. Export from Feature 1 or upload one.")
    es = from_json(json.load(open(p,"r",encoding="utf-8")))
    if not es:
        raise gr.Error("JSON has no resources.")
    GEN["entries"] = es
    lines = ["ID %d: %s | %s | t=%s" % (e.id, e.label, e.color_hex, e.threshold) for e in es]
    return es, "Loaded %d from %s:" % (len(es), os.path.basename(p)) + chr(10) + chr(10).join(lines)

def pfield(w, h, scale, octv, seed):
    pn = PerlinNoise(octaves=int(octv), seed=int(seed))
    xs = np.linspace(0, float(scale), w)
    ys = np.linspace(0, float(scale), h)
    f = np.zeros((h, w))
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            f[j,i] = pn([x, y])
    mn, mx = f.min(), f.max()
    if mx > mn:
        return (f-mn)/(mx-mn)
    return np.full_like(f, 0.5)

def domap(nz, es, base_field=None, n_bases=1, min_base_dist=30):
    order = sorted(range(len(es)), key=lambda k: es[k].threshold, reverse=True)
    g = np.full(nz.shape, order[-1], dtype=np.int32)
    for k in reversed(order):
        g[nz >= es[k].threshold] = k

    base_idx = np.full(nz.shape, -1, dtype=np.int32)

    if base_field is not None and n_bases > 0:
        h, w = base_field.shape
        flat_idx = np.argsort(base_field.ravel())[::-1]
        placed = 0
        for idx in flat_idx:
            if placed >= n_bases:
                break
            x, y = idx % w, idx // w
            if placed == 0:
                base_idx[y, x] = placed
                placed += 1
            else:
                # greedy: skip cells too close to existing bases
                existed = np.column_stack(np.where(base_idx >= 0))
                if len(existed) == 0:
                    continue
                dists = np.sqrt((existed[:, 1] - x) ** 2 + (existed[:, 0] - y) ** 2)
                if dists.min() >= min_base_dist:
                    base_idx[y, x] = placed
                    placed += 1
    return g, base_idx

def place_resources(base_idx, res_field, n_res, radius=10):
    h, w = res_field.shape
    bases = np.unique(base_idx)
    resources = []
    for base_id in bases:
        if base_id == -1:
            continue
        mask = base_idx == base_id
        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue
        cy, cx = int(ys.mean()), int(xs.mean())
        candidates = []
        for yy in range(max(0, cy-radius), min(h, cy+radius+1)):
            for xx in range(max(0, cx-radius), min(w, cx+radius+1)):
                candidates.append((res_field[yy, xx], yy, xx))
        candidates.sort(key=lambda t: t[0], reverse=True)
        count = 0
        for _, yy, xx in candidates:
            if count >= n_res:
                break
            resources.append((int(base_id), int(xx), int(yy)))
            count += 1
    return resources

def _texture_grid(g, es):
    """Per-cell RGB texture (H x W x 3 uint8) the generator renders from.

    One real pixel colour per map cell, derived from the material table,
    instead of a flat fill colour.
    """
    h, w = g.shape
    tex = np.zeros((h, w, 3), dtype=np.uint8)
    for k, e in enumerate(es):
        try:
            rgb = hex_rgb(e.color_hex)
        except Exception:
            rgb = (128, 128, 128)
        tex[g == k] = rgb
    return tex


def drawmap(g, es, ps, base_idx=None, resources=None, texture=None):
    h, w = g.shape
    if texture is not None:
        cv = np.asarray(texture, dtype=np.uint8).copy()
    else:
        cv = np.zeros((h, w, 3), dtype=np.uint8)
        for k, e in enumerate(es):
            try:
                rgb = hex_rgb(e.color_hex)
            except Exception:
                rgb = (128,128,128)
            cv[g==k] = rgb

    # Base markers
    if base_idx is not None:
        for base_id in np.unique(base_idx):
            if base_id == -1:
                continue
            mask = base_idx == base_id
            ys, xs = np.where(mask)
            if len(ys) == 0:
                continue
            cy, cx = int(ys.mean()), int(xs.mean())
            # Base marker: colored wedge on the base cell
            col = PAL[base_id % len(PAL)]
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    if abs(dx) != abs(dy) and abs(dx) + abs(dy) <= 3:
                        ny, nx = cy+dy, cx+dx
                        if 0 <= ny < cv.shape[0] and 0 <= nx < cv.shape[1]:
                            cv[ny, nx] = col

    # Resource markers
    if resources:
        for base_id, rx, ry in resources:
            col = (255, 255, 255)
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if abs(dx) != abs(dy):
                        ny, nx = ry+dy, rx+dx
                        if 0 <= ny < cv.shape[0] and 0 <= nx < cv.shape[1]:
                            cv[ny, nx] = col

    im = Image.fromarray(cv, mode="RGB")
    if ps > 1:
        im = im.resize((g.shape[1]*ps, g.shape[0]*ps), Image.NEAREST)
    return im

def add_legend(im, es, g, base_idx=None, resources=None):
    cnt = np.bincount(g.ravel(), minlength=len(es))
    tot = cnt.sum()
    out = Image.new("RGB", (im.width+260, max(im.height, 40+28*len(es))), (24,24,28))
    out.paste(im, (0,0))
    d = ImageDraw.Draw(out)
    d.text((im.width+16, 10), "Legend", fill=(255,255,255))
    for i, e in enumerate(es):
        y = 40+i*28
        try:
            rgb = hex_rgb(e.color_hex)
        except Exception:
            rgb = (128,128,128)
        d.rectangle([im.width+16, y, im.width+40, y+18], fill=rgb, outline=(255,255,255))
        pct = 100.0*cnt[i]/max(tot,1)
        d.text((im.width+48, y), "%s %.1f pct" % (e.label, pct), fill=(230,230,230))

    # Base legend
    if base_idx is not None:
        bases = np.unique(base_idx)
        for base_id in bases:
            if base_id == -1:
                continue
            y = 40 + len(es)*28 + 20 + base_id*28
            d.rectangle([im.width+16, y, im.width+40, y+18], fill=PAL[base_id % len(PAL)], outline=(255,255,255))
            d.text((im.width+48, y), "Base %d" % base_id, fill=(230,230,230))
    if resources:
        for base_id, rx, ry in resources[:8]:  # limit to 8 for legend
            y = 40 + len(es)*28 + 20 + len(bases)*28 + 20 + base_id*28
            try:
                rgb = hex_rgb(PAL[base_id % len(PAL)])
            except Exception:
                rgb = (128,128,128)
            d.ellipse([im.width+16+8, y+4, im.width+16+24, y+20], fill=(255,255,255), outline=rgb)
            d.text((im.width+16+30, y-2), "⛶ %d" % base_id, fill=(230,230,230))
    return out

def f2_generate(jf, w, h, sc, oc, sd, ps, n_bases=1, n_res=2):
    # Materials: reuse what is already loaded, otherwise read the upload.
    # NOTE: the JSON payload is *always* inspected when an upload is given,
    # so a `texture` carried by the JSON reaches the generator even when
    # materials were loaded earlier (previously jf was ignored in that case).
    es = GEN.get("entries")
    jf_payload = None
    if jf is not None and not (isinstance(jf, str) and not jf.strip()):
        try:
            _, jf_payload = _read_json_payload(jf)
        except Exception:
            jf_payload = None
    if es is None:
        es = from_json(jf_payload) if isinstance(jf_payload, dict) else []
        if not es:
            raise gr.Error("No materials. Export (Feature 1) or upload a resource_label.json.")
        GEN["entries"] = es
    w = max(8, min(int(w), 512))
    h = max(8, min(int(h), 512))
    nz = pfield(w, h, float(sc), int(oc), int(sd))

    # Base placement: low-freq noise, greedy spacing
    base_field = pfield(w, h, float(sc) * 2.0, 3, int(sd))
    g, base_idx = domap(nz, es, base_field=base_field, n_bases=n_bases, min_base_dist=30)

    # Resources: high-freq noise, clustered on each base
    res_field = pfield(w, h, float(sc) * 5.0, 7, int(sd) + 77777)
    resources = place_resources(base_idx, res_field, n_res)

    # Texture the generator renders from: if the input JSON already carries a
    # per-cell texture, use it directly; otherwise derive colours from materials.
    texture = None
    if jf_payload is not None and "texture" in jf_payload:
        raw = jf_payload["texture"]
        if raw is not None:
            try:
                texture = np.asarray(raw, dtype=np.uint8)
                if texture.ndim != 3 or texture.shape[2] != 3:
                    texture = None
            except Exception:
                texture = None
    if texture is not None and (texture.shape[0] != g.shape[0] or texture.shape[1] != g.shape[1]):
        texture = None
    if texture is None:
        texture = _texture_grid(g, es)
    GEN.update({"grid": g, "base_idx": base_idx, "resources": resources, "seed": int(sd), "texture": texture})
    im = add_legend(drawmap(g, es, int(ps), base_idx=base_idx, resources=resources, texture=texture), es, g,
                    base_idx=base_idx, resources=resources)
    cnt = np.bincount(g.ravel(), minlength=len(es))
    tot = cnt.sum()
    parts = ["%s:%.1f pct" % (es[k].label, 100.0*cnt[k]/tot) for k in range(len(es))]
    base_str = ", ".join(["B%d:%d" % (b, (base_idx==b).sum()) for b in np.unique(base_idx) if b >= 0])
    res_str = ", ".join(["B%d:%d" % (b, sum(1 for r in resources if r[0]==b)) for b in np.unique(base_idx) if b >= 0])
    st = "Seed=%s %dx%d | Bases: %s | Res: %s | " % (sd, w, h, base_str, res_str) + ", ".join(parts)
    return np.array(im), st

def f2_regen(jf, w, h, sc, oc, ps, nb=1, nr=2):
    return f2_generate(jf, w, h, sc, oc, random.randint(0, 999999), ps, nb, nr)

def f2_png():
    if GEN.get("grid") is None:
        raise gr.Error("Generate first.")
    es = GEN["entries"]
    g = GEN["grid"]
    base_idx = GEN.get("base_idx")
    resources = GEN.get("resources")
    texture = GEN.get("texture")
    im = add_legend(drawmap(g, es, 8, base_idx=base_idx, resources=resources, texture=texture), es, g,
                    base_idx=base_idx, resources=resources)
    return _export_auto("procedural_map.png", lambda p: im.save(p), "Map PNG exported.")

def f2_mat():
    if GEN.get("grid") is None:
        raise gr.Error("Generate first.")
    es = GEN["entries"]
    g = GEN["grid"]
    base_idx = GEN.get("base_idx")
    resources = GEN.get("resources")
    base_idx_arr = base_idx if base_idx is not None else np.full((0, 0), -1, dtype=np.int32)
    n_bases = int((base_idx_arr >= 0).sum())
    n_res = len(resources or [])
    pay = {"meta": {"seed": GEN["seed"], "n_bases": n_bases, "n_res": n_res},
           "materials": [asdict(e) for e in es],
           "grid_ids": g.tolist(),
           "grid_labels": [[es[int(v)].label for v in row] for row in g.tolist()],
           "base_grid": base_idx.tolist() if base_idx is not None else None,
           "resource_positions": [[int(r[0]), int(r[1]), int(r[2])] for r in (resources or [])],
           "texture": _texture_grid(g, es).tolist()}
    return _export_auto("generated_map_matrix.json",
                        lambda p: open(p, "w", encoding="utf-8").write(json.dumps(pay)),
                        "Grid JSON exported.")

def _reveal_exports():
    p = os.path.abspath(get_export_dir())
    try:
        os.startfile(p)  # type: ignore[attr-defined]
    except Exception as e:
        raise gr.Error("Could not open folder %s: %s" % (_rel(p), e))
    return "Opened:\n" + _rel(p)

def build_ui():
    with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft()) as d:
        gr.Markdown("# " + APP_TITLE + " - offline Tab1 scan, Tab2 generate")
        with gr.Tab("Feature 1 - Scan"):
            with gr.Row():
                with gr.Column():
                    im = gr.Image(type="numpy", label="Upload or paste image (click to browse, drag-drop, or Ctrl+V)",
                                  sources=["upload", "clipboard"])
                    kk = gr.Slider(2, 12, value=5, step=1, label="Colors (K)")
                    md = gr.Radio(["Pattern groups", "Object lumps", "Color lumps"], value="Pattern groups",
                        label="Segment mode")
                    mp = gr.Slider(3, 15, value=7, step=2, label="Cleanup size (bigger = more solid lumps)")
                    ma = gr.Slider(0, 5000, value=400, step=50, label="Min lump area px (pattern/object mode)")
                    mg = gr.Slider(0.3, 3.0, value=1.0, step=0.1,
                        label="Pattern merge strength (higher = fewer, bigger groups)")
                    sb = gr.Button("Scan Image", variant="primary")
                with gr.Column():
                    so = gr.HTML(value=_empty_view(), label="Segmented (tint + real-map overlay)")
                    ov = gr.Slider(0.0, 1.0, value=0.35, step=0.01,
                        label="Real-map overlay alpha (0 = pure tint, 1 = real map)",
                        elem_id="ovslider")
                    st = gr.Textbox(label="Status", interactive=False)
            tb = gr.Dataframe(headers=["Segment ID", "Mean Color", "Pixel pct", "Label", "Display Color", "Threshold", "Swatch"],
                datatype=["number", "str", "number", "str", "str", "number", "html"],
                col_count=(7, "fixed"), interactive=True, wrap=True, label="Segments (hover any row for spotlight minimap)",
                elem_id="segtable")
            with gr.Row():
                eb = gr.Button("💾 Save resource_label.json", variant="primary")
            with gr.Row():
                jo = gr.Textbox(label="Saved file path", interactive=False, show_copy_button=True)
            with gr.Row():
                reveal1 = gr.Button("📁 Open export folder")
            sb.click(f1_scan, [im, kk, md, mp, ma, ov, mg], [so, tb, st])
            d.load(None, None, None, js="""() => {
if (window.__segHover) return; window.__segHover = true;
const css = document.createElement('style');
css.textContent = '#segtable td{overflow:visible !important;} #segtable .table-wrap{overflow:visible !important;}' +
' #rowtip{position:fixed;z-index:9999;display:none;pointer-events:none;background:#111;border:1px solid #666;border-radius:8px;padding:6px;}' +
' #rowtip img{width:180px;height:180px;image-rendering:pixelated;display:block;}' +
' #rowtip .cap{color:#eee;font-size:12px;margin-bottom:4px;}' +
' #segtable tbody tr{transition:background 0.12s;} #segtable tbody tr.hlit{background:rgba(255,255,0,0.12) !important;}';
document.head.appendChild(css);
const tip = document.createElement('div'); tip.id = 'rowtip';
tip.innerHTML = '<div class=cap></div><img/>';
document.body.appendChild(tip);
const timg = tip.querySelector('img'), tcap = tip.querySelector('.cap');
const tbl = () => document.querySelector('#segtable');
const idOf = (tr) => {
  const sw = tr.querySelector('.swwrap');
  if (sw && sw.dataset && sw.dataset.gid) return sw.dataset.gid;
  const tds = tr.querySelectorAll('td');
  if (tds.length) return (tds[0].innerText || '').trim();
  return '';
};
const show = (tr, x, y) => {
  const gid = idOf(tr);
  const sw = tr.querySelector('.swwrap');
  const src = sw ? sw.dataset.thumb : null;
  if (!src) return;
  tcap.textContent = 'ID ' + gid + ' spotlight (row hover)';
  if (timg.src !== src) timg.src = src;
  tip.style.display = 'block';
  const pad = 16, W = 200, H = 230;
  tip.style.left = Math.min(x + pad, window.innerWidth - W) + 'px';
  tip.style.top = Math.min(y + pad, window.innerHeight - H) + 'px';
};
const hide = () => { tip.style.display = 'none'; };
document.addEventListener('mousemove', e => {
  const t = tbl(); if (!t) { hide(); return; }
  const tr = e.target.closest ? e.target.closest('#segtable tbody tr') : null;
  t.querySelectorAll('tbody tr.hlit').forEach(r => { if (r !== tr) r.classList.remove('hlit'); });
  if (!tr) { hide(); return; }
  tr.classList.add('hlit');
  show(tr, e.clientX, e.clientY);
});
document.addEventListener('mouseleave', hide, true);
}""")
            ov.input(None, [ov], None,
                js="(a) => { var lay = document.getElementById('reallay');"
                   " if (lay) lay.style.opacity = (parseFloat(a) || 0).toString(); }")
            eb.click(f1_export, [tb], [jo])
            reveal1.click(_reveal_exports, None, [jo])
        with gr.Tab("Feature 2 - Generator"):
            with gr.Row():
                with gr.Column():
                    jf = gr.File(label="resource_label.json", file_types=[".json"])
                    lb = gr.Button("Load Materials")
                    mi = gr.Textbox(label="Materials", lines=6, interactive=False)
                    wi = gr.Slider(16, 512, value=128, step=8, label="Width")
                    hi = gr.Slider(16, 512, value=128, step=8, label="Height")
                    sc = gr.Slider(0.5, 20.0, value=5.0, step=0.5, label="Scale")
                    oc = gr.Slider(1, 8, value=4, step=1, label="Octaves")
                    sd = gr.Number(value=0, label="Seed", precision=0)
                    ps = gr.Slider(1, 16, value=4, step=1, label="Pixel scale")
                    nb = gr.Slider(1, 8, value=1, step=1, label="Bases count")
                    nr = gr.Slider(1, 4, value=2, step=1, label="Resources per base")
                    with gr.Row():
                        gb = gr.Button("Generate", variant="primary")
                        rb = gr.Button("Regenerate")
                with gr.Column():
                    mo = gr.Image(label="Map plus legend")
                    ss = gr.Textbox(label="Stats", interactive=False)
                    with gr.Row():
                        pb = gr.Button("💾 Save PNG")
                        mb = gr.Button("💾 Save grid JSON")
                    with gr.Row():
                        po = gr.Textbox(label="Saved PNG path", interactive=False, show_copy_button=True)
                        mo2 = gr.Textbox(label="Saved grid JSON path", interactive=False, show_copy_button=True)
                    with gr.Row():
                        reveal2 = gr.Button("📁 Open export folder")
            lb.click(lambda f: load_entries(f)[1], [jf], [mi])
            gb.click(f2_generate, [jf, wi, hi, sc, oc, sd, ps, nb, nr], [mo, ss])
            rb.click(f2_regen, [jf, wi, hi, sc, oc, ps, nb, nr], [mo, ss])
            pb.click(f2_png, None, [po])
            mb.click(f2_mat, None, [mo2])
            reveal2.click(_reveal_exports, None, [mo2])
        with gr.Tab("⚙ Settings"):
            gr.Markdown("### Export folder\nAll Save buttons write here automatically. "
                        "Pick a folder once - it is remembered in `settings.json`.")
            with gr.Row():
                ex_dir = gr.Textbox(label="Export folder", value=_rel(get_export_dir()), interactive=False,
                                    show_copy_button=True, scale=4)
                ex_open = gr.Button("📁 Open", scale=1)
            with gr.Row():
                ex_pick = gr.File(label="Browse: pick ANY file inside the folder you want",
                                  file_count="single", type="filepath")
                ex_save = gr.Button("✔ Use this folder", variant="primary")
            ex_msg = gr.Textbox(label="Status", interactive=False)
            ex_open.click(_reveal_exports, None, [ex_msg])

            def _use_folder(p):
                if not p:
                    raise gr.Error("Browse and pick any file inside the target folder first.")
                folder = p if os.path.isdir(p) else os.path.dirname(os.path.abspath(p))
                msg = set_export_dir(folder)
                return msg, _rel(folder)

            ex_save.click(_use_folder, [ex_pick], [ex_msg, ex_dir])
    return d

def launch():
    demo = build_ui()
    threading.Thread(target=lambda: demo.launch(server_name="127.0.0.1", server_port=GRADIO_PORT,
        inbrowser=False, show_error=True, prevent_thread_lock=True, quiet=True), daemon=True).start()
    url = "http://127.0.0.1:%d" % GRADIO_PORT
    import urllib.request
    for _ in range(60):
        try:
            urllib.request.urlopen(url, timeout=2)
            break
        except Exception:
            time.sleep(0.5)
    try:
        import webview
        webview.create_window(APP_TITLE, url, width=1280, height=860)
        webview.start()
    except Exception as e:
        print("webview fail, open " + url + " err=" + str(e))
        webbrowser.open(url)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    launch()
