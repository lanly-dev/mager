import json, os, random, sys, threading, time, webbrowser
from contextlib import contextmanager
from dataclasses import dataclass, asdict
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
    except Exception as e:
        print("mager: could not load settings.json (%s); using defaults" % e, file=sys.stderr)
        return {}

def _save_settings(s):
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2)
    except Exception as e:
        print("mager: could not save settings.json (%s)" % e, file=sys.stderr)

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
GEN = {"entries": None, "grid": None, "seed": None, "height": None,
       "base_idx": None, "resources": None, "texture": None}
SWAP = {"orig": None, "out": None, "prev": [], "remaps": [], "entries": None}
# Guards SCAN/GEN/SWAP against concurrent Gradio handler threads mutating
# shared state mid-flight. Use via the _state_locked() context manager below.
STATE_LOCK = threading.Lock()

@contextmanager
def _state_locked():
    """Non-blocking guard for SCAN/GEN/SWAP mutations.

    Gradio runs each handler in its own thread; a second click while one is
    running gets gr.Error("busy...") instead of interleaving mutations or
    deadlocking the UI thread."""
    if not STATE_LOCK.acquire(blocking=False):
        raise gr.Error("Busy - another operation is running. Try again.")
    try:
        yield
    finally:
        STATE_LOCK.release()

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
    texture_png: str = ""  # data-URL PNG exemplar cropped from the scanned reference image

def to_json(es):
    return {"meta": {"app": "MAGER", "version": 1},
            "resources": [asdict(e) for e in es]}

def from_json(data):
    out = []
    for r in data.get("resources") or data.get("materials") or []:
        try:
            rid = int(r["id"])
            out.append(Res(rid, str(r.get("label","seg")),
                str(r.get("color_hex","#808080")),
                str(r.get("mean_color_hex","#808080")),
                float(r.get("coverage",0)), float(r.get("threshold",0)),
                str(r.get("texture_png","") or "")))
        except (KeyError, ValueError, TypeError) as e:
            # Skip malformed rows instead of tracebacking; callers raise a
            # clean gr.Error when zero valid resources remain.
            print("mager: skipping malformed resource row (%s): %r" % (e, r), file=sys.stderr)
            continue
    return out

def _exemplars_from_payload(payload):
    """Decode {seg_id: exemplar RGB} carried by a resource_label.json payload."""
    out = {}
    try:
        rows = payload.get("resources") or payload.get("materials") or []
        for r in rows:
            try:
                gid = int(r.get("id", -1))
            except Exception:
                continue
            arr = _from_b64_png(r.get("texture_png",""))
            if arr is not None:
                out[gid] = arr
    except Exception:
        pass
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

def _b64_png(img):
    """Encode an RGB uint8 array as a data-URL PNG (compact texture exemplar)."""
    import base64, io
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img).astype(np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def _from_b64_png(s):
    """Decode a data-URL (or raw base64) PNG back to an RGB uint8 array. None on failure."""
    try:
        import base64, io
        if not s or not isinstance(s, str):
            return None
        if "," in s:
            s = s.split(",", 1)[1]
        raw = base64.b64decode(s)
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        return np.array(im)
    except Exception:
        return None

def _exemplar_crop(orig, label_map, gid, size=48):
    """Crop a size×size RGB exemplar of segment gid from the scanned reference image.

    Picks the largest connected blob of that segment, crops around its centroid,
    and falls back to a mean-colour tile when the crop would be empty.
    """
    try:
        size = max(8, min(int(size), 128))
        m = (np.asarray(label_map) == int(gid))
        if not m.any():
            return None
        mm = m.astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            M = cv2.moments(c)
            if M.get("m00", 0) > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                ys, xs = np.where(m)
                cy, cx = int(ys.mean()), int(xs.mean())
        else:
            ys, xs = np.where(m)
            cy, cx = int(ys.mean()), int(xs.mean())
        h, w, _ = orig.shape
        x0 = max(0, min(w - size, cx - size // 2))
        y0 = max(0, min(h - size, cy - size // 2))
        crop = orig[y0:y0 + size, x0:x0 + size]
        if crop.size == 0:
            return None
        if crop.shape[0] != size or crop.shape[1] != size:
            crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(crop)
    except Exception:
        return None

def _exemplar_rgb(e):
    """Decode a Res material's embedded texture_png exemplar to RGB array. None on failure."""
    try:
        return _from_b64_png(getattr(e, "texture_png", ""))
    except Exception:
        return None

def _scan_exemplars(size=48):
    """Build {seg_id: exemplar RGB array} from the last Feature 1 scan.

    Falls back to a mean-colour tile per segment when no reference pixels exist.
    """
    out = {}
    orig = SCAN.get("orig")
    lmap = SCAN.get("obj_map")
    info = SCAN.get("info") or []
    for s in info:
        gid = int(s.get("id", -1))
        ex = None
        if orig is not None and lmap is not None:
            try:
                if lmap.shape[:2] != orig.shape[:2]:
                    lm = cv2.resize(lmap.astype(np.int32),
                                    (orig.shape[1], orig.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
                else:
                    lm = lmap
                ex = _exemplar_crop(orig, lm, gid, size=size)
            except Exception:
                ex = None
        if ex is None:
            try:
                rgb = hex_rgb(s.get("mean", "#808080"))
            except Exception:
                rgb = (128, 128, 128)
            ex = np.full((size, size, 3), rgb, dtype=np.uint8)
        out[gid] = np.ascontiguousarray(ex)
    return out

def _tile_texture(g, exemplars, es, seed=0, jitter=0):
    """Per-pixel texture for map grid g by tiling each segment's exemplar.

    Each cell samples its segment exemplar with a deterministic per-cell offset
    (seeded by `seed`), so large regions show the reference grain instead of a
    flat fill. `jitter` (0..64) randomly perturbs sampled pixels for variety.
    """
    h, w = (int(g.shape[0]), int(g.shape[1]))
    tex = np.zeros((h, w, 3), dtype=np.uint8)
    rng = np.random.default_rng(int(seed))
    try:
        jit = max(0, min(int(jitter), 64))
    except Exception:
        jit = 0
    for k, e in enumerate(es):
        mask = (g == k)
        if not mask.any():
            continue
        ex = None
        try:
            ex = exemplars.get(int(getattr(e, "id", k)), exemplars.get(k))
        except Exception:
            ex = None
        if ex is None:
            try:
                rgb = hex_rgb(e.color_hex)
            except Exception:
                rgb = (128, 128, 128)
            tex[mask] = rgb
            continue
        ex = np.asarray(ex, dtype=np.uint8)
        if ex.ndim != 3 or ex.shape[2] != 3 or ex.shape[0] < 1 or ex.shape[1] < 1:
            try:
                rgb = hex_rgb(e.color_hex)
            except Exception:
                rgb = (128, 128, 128)
            tex[mask] = rgb
            continue
        eh, ew, _ = ex.shape
        oy = int(rng.integers(0, max(eh, 1)))
        ox = int(rng.integers(0, max(ew, 1)))
        ys, xs = np.where(mask)
        ty = (ys + oy) % eh
        tx = (xs + ox) % ew
        tex[ys, xs] = ex[ty, tx]
    if jit > 0:
        nz = rng.integers(-jit, jit + 1, size=tex.shape, dtype=np.int16)
        tex = np.clip(tex.astype(np.int16) + nz, 0, 255).astype(np.uint8)
    return tex

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
    SCAN["obj_map"] = lm.astype(np.int32)
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

def _to_b64(img):
    import base64, io
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def _empty_view():
    return ("<div style='color:#888;padding:20px;text-align:center'>"
            "Run <b>Scan Image</b> - tint + real map appear here, slider blends instantly.</div>")

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

# Unique overlay ids across re-renders: a second scan used to emit a second
# id="reallay", leaving duplicate ids in the DOM for the slider JS to trip on.
_blend_uid = 0

def _blend_view(tint_b64, real_b64, alpha):
    global _blend_uid
    _blend_uid += 1
    uid = "reallay%d" % _blend_uid
    try:
        a = max(0.0, min(1.0, float(alpha)))
    except Exception:
        a = 0.35
    return (
        "<div style='position:relative;width:100%;background:#111'>"
        "<img src='" + tint_b64 + "' style='display:block;width:100%;image-rendering:pixelated'/>" +
        "<img id='" + uid + "' src='" + real_b64 + "' style='position:absolute;inset:0;width:100%;height:100%;object-fit:fill;pointer-events:none;opacity:" + str(a) + "'/>" +
        "<script>(function(){var s=document.querySelector('#ovslider input[type=range]');"
        "var lay=document.getElementById('" + uid + "');"
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
    es = []
    for i, s in enumerate(info):
        pb = PAL[s["id"] % len(PAL)]
        hx = bgr_hex((int(pb[0]), int(pb[1]), int(pb[2])))
        sw = _swatch_popup(s["id"], hx, vis, image)
        rows.append([s["id"], s["mean"], round(s["cov"] * 100, 2),
                     NAMES[i % len(NAMES)], hx, th[i], sw])
        # Mirror f1_export's Res construction so Feature 3 ("reuse Scan")
        # works straight from memory without an export round-trip.
        es.append(Res(int(s["id"]), NAMES[i % len(NAMES)], hx,
                      str(s["mean"]), round(float(s["cov"]), 5), float(th[i])))
    with _state_locked():
        SCAN["info"] = info
        SCAN["vis"] = vis
        SCAN["orig"] = image
        # Attach real-pixel exemplars (same as f1_export) so the texture-swap
        # tab renders true texture, not flat colour, from scan materials.
        try:
            _ex = _scan_exemplars(size=48)
            for _e in es:
                _arr = _ex.get(int(_e.id))
                if _arr is not None:
                    _e.texture_png = _b64_png(_arr)
        except Exception as e:
            print("mager: scan exemplar attach failed (%s)" % e, file=sys.stderr)
        SCAN["entries"] = es
    view = _blend_view(_to_b64(vis), _to_b64(image), alpha)
    return view, rows, "Found %d segments (%s). Hover a swatch for its minimap popup." % (len(info), mode)

def _save_export(name, write):
    p = os.path.join(get_export_dir(), name)
    write(p)
    return p

def _export_auto(name, write_file, kind):
    """Save straight to the configured export folder. No dialogs."""
    import datetime
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # usec: no same-second collisions
    base, ext = os.path.splitext(name)
    p = _save_export("%s_%s%s" % (base, stamp, ext), write_file)
    return "OK - %s\nSaved: %s" % (kind, _rel(p))

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
    # Attach a real-pixel exemplar per segment (cropped from the scanned
    # reference image) so Feature 2 renders texture, not just flat colour.
    try:
        _ex = _scan_exemplars(size=48)
        for _e in es:
            _arr = _ex.get(int(_e.id))
            if _arr is not None:
                _e.texture_png = _b64_png(_arr)
    except Exception as e:
        print("mager: export exemplar attach failed (%s)" % e, file=sys.stderr)
    pay = to_json(es)
    def _write(p):
        open(p, "w", encoding="utf-8").write(json.dumps(pay, indent=2))
    # NOTE: the export-dir copy (via _export_auto) is the only one kept; the
    # old second copy inside the app folder was removed (stray untracked file).
    return _export_auto("resource_label.json", _write, "resource_label.json exported.")
def load_entries(fobj):
    dflt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resource_label.json")
    p = fobj if fobj else (dflt if os.path.exists(dflt) else None)
    if not p or not os.path.exists(p):
        raise gr.Error("No resource_label.json. Export from Feature 1 or upload one.")
    try:
        payload = json.load(open(p, "r", encoding="utf-8"))
    except Exception as e:
        raise gr.Error("Could not parse %s: %s" % (os.path.basename(p), e))
    es = from_json(payload)
    if not es:
        raise gr.Error("JSON has no resources.")
    GEN["entries"] = es
    lines = ["ID %d: %s | %s | t=%s" % (e.id, e.label, e.color_hex, e.threshold) for e in es]
    return es, "Loaded %d from %s:" % (len(es), os.path.basename(p)) + chr(10) + chr(10).join(lines)

def pfield(w, h, scale, octv, seed):
    # The perlin-noise package evaluates in pure Python (~22us/call), so a
    # 512x512 field costs ~6s. Hoisting attribute lookups was measured at
    # <1% gain - the per-call overhead inside the package dominates and can't
    # be removed while staying bit-exact.
    # Tradeoff: fields larger than 256x256 cells are evaluated at half
    # resolution and cubically upsampled (~4x fewer calls). Values differ
    # slightly from a full-resolution evaluation, so the same seed yields a
    # slightly different map at large sizes - but results stay deterministic
    # per (w, h, scale, octaves, seed), which is all the UI promises.
    div = 1
    while (w // div) * (h // div) > 65536 and div < 8:
        div *= 2
    cw, ch = max(1, w // div), max(1, h // div)
    pn = PerlinNoise(octaves=int(octv), seed=int(seed))
    noise = pn.noise  # hoisted: skip __call__ dispatch per cell (same values)
    xs = np.linspace(0, float(scale), cw)
    ys = np.linspace(0, float(scale), ch)
    f = np.zeros((ch, cw))
    for j, y in enumerate(ys):
        row = f[j]
        for i, x in enumerate(xs):
            row[i] = noise([x, y])
    if div > 1:
        f = cv2.resize(f, (w, h), interpolation=cv2.INTER_CUBIC)
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
        # Greedy highest-first placement with mask-based exclusion: when a
        # base is placed, a precomputed disk marks every cell within
        # min_base_dist as blocked, so each candidate costs O(1) instead of
        # measuring distances to all placed bases (old code: O(cells x bases)
        # with per-cell numpy overhead). A cell stays eligible exactly when
        # its distance to every placed base is >= min_base_dist, matching the
        # old `dists.min() >= min_base_dist` rule cell-for-cell.
        by_score = np.argsort(base_field.ravel())[::-1]
        blocked = np.zeros((h, w), dtype=bool)
        r = int(min_base_dist)
        _dy, _dx = np.mgrid[-r:r + 1, -r:r + 1]
        disk = (_dx * _dx + _dy * _dy) < r * r
        placed = 0
        for idx in by_score:
            if placed >= n_bases:
                break
            x, y = int(idx % w), int(idx // w)
            if blocked[y, x]:
                continue
            base_idx[y, x] = placed
            placed += 1
            y0, y1 = max(0, y - r), min(h, y + r + 1)
            x0, x1 = max(0, x - r), min(w, x + r + 1)
            blocked[y0:y1, x0:x1] |= disk[y0 - (y - r):y1 - (y - r),
                                          x0 - (x - r):x1 - (x - r)]
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
            # Base marker: colored wedge on the base cell.
            # PAL is BGR but cv is RGB (Image.fromarray(..., "RGB")) -> unpack.
            _b, _g, _r = PAL[base_id % len(PAL)]
            col = (_r, _g, _b)
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
    # `bases` is defined unconditionally: the resource-row layout below uses
    # len(bases), which used to NameError when base_idx was None.
    bases = np.unique(base_idx) if base_idx is not None else np.array([-1])
    base_ids = [int(b) for b in bases if b != -1]
    res_rows = list((resources or [])[:8])  # legend shows at most 8
    # Size the canvas to fit every drawn row (materials + base rows +
    # resource rows + padding). The old height ignored base/resource rows and
    # silently clipped them on small maps.
    need = 40 + 28 * len(es)
    if base_ids:
        need = max(need, 40 + len(es) * 28 + 20 + max(base_ids) * 28 + 28)
    if res_rows:
        need = max(need, 40 + len(es) * 28 + 20 + len(bases) * 28 + 20
                   + max(int(r[0]) for r in res_rows) * 28 + 28)
    need += 12  # bottom padding
    out = Image.new("RGB", (im.width+260, max(im.height, need)), (24,24,28))
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

    # Base legend (PAL is BGR -> unpack to RGB for PIL)
    for base_id in base_ids:
        y = 40 + len(es)*28 + 20 + base_id*28
        _b, _g, _r = PAL[base_id % len(PAL)]
        d.rectangle([im.width+16, y, im.width+40, y+18], fill=(_r, _g, _b), outline=(255,255,255))
        d.text((im.width+48, y), "Base %d" % base_id, fill=(230,230,230))
    if res_rows:
        for base_id, rx, ry in res_rows:
            y = 40 + len(es)*28 + 20 + len(bases)*28 + 20 + base_id*28
            # PAL is BGR -> unpack to RGB for PIL (hex_rgb takes a string,
            # so passing the tuple always raised and fell back to gray)
            _b, _g, _r = PAL[base_id % len(PAL)]
            rgb = (_r, _g, _b)
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
        with _state_locked():
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

    # Texture the generator renders from (priority order):
    #  1. full per-cell "texture" carried by a generated_map_matrix.json upload
    #     (exact pixels, same H×W) -> use directly;
    #  2. per-material "texture_png" exemplars cropped from the scanned
    #     reference image (resource_label.json) -> tile per cell so regions
    #     show real grain instead of a flat fill;
    #  3. flat material colours (old fallback).
    texture = None
    if isinstance(jf_payload, dict) and "texture" in jf_payload:
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
        # Tile real reference pixels carried by the materials JSON.
        # Exemplars merge by material id: the fresh upload wins per id, but
        # ids missing from the upload keep the already-loaded (GEN) exemplar
        # for that id, so a partial or id-mismatched JSON degrades gracefully
        # instead of silently dropping those regions to flat colour. Last
        # resort is the live Feature 1 scan, then flat material colours.
        exemplars = {}
        try:
            for _e in (es or []):
                _arr = _from_b64_png(getattr(_e, "texture_png", ""))
                if _arr is not None:
                    exemplars[int(getattr(_e, "id", -1))] = _arr
        except Exception:
            exemplars = {}
        if isinstance(jf_payload, dict):
            exemplars.update(_exemplars_from_payload(jf_payload))
        if not exemplars:
            try:
                exemplars = _scan_exemplars(size=48)
            except Exception:
                exemplars = {}
        if exemplars:
            texture = _tile_texture(g, exemplars, es, seed=int(sd))
        else:
            texture = _texture_grid(g, es)
    with _state_locked():
        GEN.update({"grid": g, "height": nz, "base_idx": base_idx, "resources": resources,
                    "seed": int(sd), "texture": texture})
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
    es = GEN.get("entries") or []
    g = GEN["grid"]
    base_idx = GEN.get("base_idx")
    resources = GEN.get("resources")
    # Grid cells hold positional indices into `es`. Guard against a stale
    # grid (generated) vs reloaded materials (different length) mismatch:
    # label out-of-range cells "?" and warn instead of IndexError-tracebacking.
    _n = len(es)
    _bad = [0]
    def _lab(v):
        try:
            _i = int(v)
        except (TypeError, ValueError):
            _i = -1
        if 0 <= _i < _n:
            return es[_i].label
        _bad[0] += 1
        return "?pos%d" % _i
    base_idx_arr = base_idx if base_idx is not None else np.full((0, 0), -1, dtype=np.int32)
    n_bases = int((base_idx_arr >= 0).sum())
    n_res = len(resources or [])
    _tex = GEN.get("texture")
    try:
        _tex_arr = np.asarray(_tex, dtype=np.uint8) if _tex is not None else None
    except Exception:
        _tex_arr = None
    if _tex_arr is None or _tex_arr.shape[:2] != g.shape[:2]:
        _tex_arr = _texture_grid(g, es)  # flat-colour fallback only
    pay = {"meta": {"seed": GEN["seed"], "n_bases": n_bases, "n_res": n_res},
           "materials": [asdict(e) for e in es],
           "grid_ids": g.tolist(),
           "grid_labels": [[_lab(v) for v in row] for row in g.tolist()],
           "base_grid": base_idx.tolist() if base_idx is not None else None,
           "resource_positions": [[int(r[0]), int(r[1]), int(r[2])] for r in (resources or [])],
           "texture": _tex_arr.tolist()}
    msg = _export_auto("generated_map_matrix.json",
                       lambda p: open(p, "w", encoding="utf-8").write(json.dumps(pay)),
                       "Grid JSON exported.")
    if _bad[0]:
        msg += ("\nWARNING: %d grid cell(s) reference material positions outside the "
                "loaded materials list (map was generated from different materials?) - "
                "regenerate before exporting." % _bad[0])
    return msg

# ---------- feature 3: texture & material swap engine ----------
def _swap_mask(img_rgb, target_hex, tol):
    try:
        tr, tg, tb = hex_rgb(target_hex)
    except Exception:
        tr, tg, tb = (128, 128, 128)
    d = np.sqrt(((img_rgb.astype(np.float32) - np.array([tr, tg, tb], np.float32)) ** 2).sum(-1))
    m = np.clip(1.0 - d / max(float(tol), 1.0), 0.0, 1.0)
    return (m * 255.0 + 0.5).astype(np.uint8)

def _resample_exemplar(tex, W, H, tile=1.0, seed=0):
    t = np.asarray(tex, dtype=np.uint8)
    if t.ndim != 3 or t.size == 0:
        return np.zeros((H, W, 3), np.uint8)
    th, tw = t.shape[:2]
    sc = max(float(tile), 0.05)
    nw, nh = max(1, int(round(tw * sc))), max(1, int(round(th * sc)))
    t = np.asarray(Image.fromarray(t[:, :, :3], "RGB").resize((nw, nh), Image.BICUBIC), np.uint8)
    th, tw = t.shape[:2]
    oy, ox = (int(seed * 13) % max(th, 1)), (int(seed * 29) % max(tw, 1))
    yy, xx = np.mgrid[0:H, 0:W]
    return t[(yy + oy) % th, (xx + ox) % tw]

def _mask_blend(base_rgb, tex_rgb, mask_u8, feather=1, strength=1.0):
    m = mask_u8.astype(np.float32) / 255.0
    if feather > 0:
        k = int(feather) * 2 + 1
        m = cv2.GaussianBlur(m, (k, k), 0)
    m = np.clip(m * float(strength), 0.0, 1.0)[..., None]
    return (base_rgb.astype(np.float32) * (1.0 - m)
            + tex_rgb.astype(np.float32) * m + 0.5).astype(np.uint8)

def f3_upload(img, jf):
    if img is None:
        raise gr.Error("Upload a map image first.")
    rgb = np.asarray(Image.fromarray(
        img.astype(np.uint8) if isinstance(img, np.ndarray) else img).convert("RGB"), np.uint8)
    es = None
    if jf is not None:
        try:
            es, _ = _read_json_payload(jf)
        except Exception:
            es = None
    if es is None:
        es = GEN.get("entries") or SCAN.get("entries")
    if not es:
        raise gr.Error("Load a resource_label.json too (upload it here, or Scan/Generate first).")
    SWAP.update(orig=rgb, out=rgb.copy(), prev=[], remaps=[], entries=es)
    names = ["%d:%s" % (e.id, e.label) for e in es]
    return (Image.fromarray(rgb, "RGB"), gr.update(choices=names, value=[]),
            gr.update(choices=names, value=names[0] if names else None),
            "Loaded %dx%d + %d materials." % (rgb.shape[1], rgb.shape[0], len(es)))

def f3_apply(targets, repl, tol=90.0, feather=1, strength=1.0, tile=1.0):
    if SWAP.get("orig") is None:
        raise gr.Error("Upload a map image first (Feature 3).")
    es = SWAP.get("entries") or GEN.get("entries") or SCAN.get("entries")
    if not es:
        raise gr.Error("No materials loaded.")
    if not targets:
        raise gr.Error("Pick at least one target material.")
    try:
        rid = int(str(repl).split(":")[0])
    except Exception:
        raise gr.Error("Pick a replacement texture.")
    src = next((e for e in es if e.id == rid), None)
    if src is None:
        raise gr.Error("Replacement material not found.")
    tex = _exemplar_rgb(src)
    if tex is None:
        tex = np.full((48, 48, 3), hex_rgb(src.color_hex), np.uint8)
    base = SWAP.get("out", SWAP["orig"]).copy()
    H, W = base.shape[:2]
    layer = _resample_exemplar(tex, W, H, tile=tile, seed=rid)
    tids = []
    for t in targets:
        try:
            tids.append(int(str(t).split(":")[0]))
        except Exception:
            pass
    combined = np.zeros((H, W), np.uint8)
    for tid in tids:
        tgt = next((e for e in es if e.id == tid), None)
        if tgt is None:
            continue
        combined = np.maximum(combined, _swap_mask(base, tgt.color_hex, tol))
    if int(combined.max()) == 0:
        return Image.fromarray(base, "RGB"), "No pixels matched (raise tolerance)."
    out = _mask_blend(base, layer, combined, feather=int(feather), strength=strength)
    with _state_locked():
        _prev = SWAP.setdefault("prev", [])
        _prev.append(base.copy())
        del _prev[:-20]  # cap undo history at 20 full-res frames
        SWAP["out"] = out
        SWAP.setdefault("remaps", []).append({"targets": tids, "replacement": rid})
    cov = 100.0 * (combined > 0).sum() / combined.size
    return Image.fromarray(out, "RGB"), "Swapped %s -> %s : %.1f%% pixels." % (
        ",".join(str(i) for i in tids), src.label, cov)

def f3_undo():
    with _state_locked():
        if SWAP.get("prev"):
            SWAP["out"] = SWAP["prev"].pop()
            if SWAP.get("remaps"):
                SWAP["remaps"].pop()
            msg = "Undid last swap."
        elif SWAP.get("orig") is not None:
            SWAP["out"] = SWAP["orig"].copy()
            SWAP["remaps"] = []
            msg = "Reset to original."
        else:
            raise gr.Error("Nothing to undo.")
        out = SWAP["out"]
    return Image.fromarray(out, "RGB"), msg

def f3_save():
    if SWAP.get("out") is None:
        raise gr.Error("Nothing to save yet.")
    d = get_export_dir()
    ts = time.strftime("%Y%m%d_%H%M%S") + "_%06d" % int((time.time() % 1) * 1e6)
    p = os.path.join(d, "swapped_map_%s.png" % ts)
    Image.fromarray(SWAP["out"], "RGB").save(p)
    jp = os.path.join(d, "swapped_remaps_%s.json" % ts)
    with open(jp, "w", encoding="utf-8") as f:
        json.dump({"meta": {"app": "MAGER", "version": 1}, "remaps": SWAP.get("remaps", [])}, f, indent=2)
    return "Saved:\n" + _rel(p) + "\n" + _rel(jp)

# ---------- feature 4: generic game map exporter ----------
def f4_export(fmt):
    g = GEN.get("grid")
    if g is None:
        raise gr.Error("Generate a map first (Feature 2).")
    es = GEN.get("entries") or []
    nz = GEN.get("height")
    if nz is None:
        nz = np.full_like(g, 0.5, dtype=np.float32)
    d = get_export_dir()
    ts = time.strftime("%Y%m%d_%H%M%S") + "_%06d" % int((time.time() % 1) * 1e6)
    outs = []
    if "a) 2D grid matrix JSON" in fmt:
        p = os.path.join(d, "export_grid_%s.json" % ts)
        payload = {"meta": {"app": "MAGER", "version": 1, "kind": "grid_matrix",
                            "seed": GEN.get("seed"), "w": int(g.shape[1]), "h": int(g.shape[0])},
                   "materials": [{"id": e.id, "label": e.label} for e in es],
                   "grid": g.astype(int).tolist()}
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        outs.append(p)
    if "b) 16-bit heightmap PNG" in fmt:
        p = os.path.join(d, "export_height_%s.png" % ts)
        cv2.imwrite(p, (np.clip(nz, 0, 1) * 65535.0 + 0.5).astype(np.uint16))
        outs.append(p)
    if "c) RGBA splatmap PNG" in fmt:
        p = os.path.join(d, "export_splat_%s.png" % ts)
        k = min(4, len(es))
        spl = np.zeros((g.shape[0], g.shape[1], 4), np.uint8)
        for i in range(k):
            spl[:, :, i] = np.where(g == i, 255, 0).astype(np.uint8)
        Image.fromarray(spl, "RGBA").save(p)
        outs.append(p)
    if "d) object/resource placement JSON" in fmt:
        p = os.path.join(d, "export_objects_%s.json" % ts)
        objs = [{"id": int(b), "x": int(x), "y": int(y)} for (b, x, y) in (GEN.get("resources") or [])]
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"meta": {"app": "MAGER", "version": 1}, "objects": objs}, f, indent=2)
        outs.append(p)
    if not outs:
        raise gr.Error("Tick at least one export format.")
    return "Exported:\n" + "\n".join(_rel(p) for p in outs)

# ---------- feature 4: map-spec export (agent-readable contract) ----------
# The spec is the hinge everything hangs on: reference extraction outputs it,
# the generator consumes it, agents write it, games render it. Schema lives at
# spec/map-spec.schema.json (draft-07, game-agnostic core + per-game profiles).
_SPEC_SCHEMA_PATH = os.path.join(HERE, "spec", "map-spec.schema.json")
_spec_schema_cache = None

def _load_spec_schema():
    global _spec_schema_cache
    if _spec_schema_cache is None:
        try:
            with open(_SPEC_SCHEMA_PATH, "r", encoding="utf-8") as f:
                _spec_schema_cache = json.load(f)
        except Exception as e:
            raise gr.Error("Cannot read spec schema %s: %s" % (_rel(_SPEC_SCHEMA_PATH), e))
    return _spec_schema_cache

def validate_spec(spec):
    """Validate a map-spec dict against spec/map-spec.schema.json.

    Raises gr.Error with a readable message on failure. Callers must validate
    BEFORE writing anything: on failure nothing is written."""
    try:
        import jsonschema
    except ImportError:
        raise gr.Error("The 'jsonschema' package is not installed. Run: pip install -r requirements.txt")
    schema = _load_spec_schema()
    try:
        jsonschema.validate(instance=spec, schema=schema)
    except Exception as e:  # ValidationError (has .message/.path); be defensive
        msg = str(getattr(e, "message", e))
        path = list(getattr(e, "path", None) or [])
        loc = ("/" + "/".join(str(p) for p in path)) if path else ""
        raise gr.Error("Map spec invalid%s: %s" % (loc, msg))

def _cell_to_world(i, j, w, h, size=600.0):
    # Cell (i, j) of a w×h grid -> world units, origin at map center.
    # Schema: x right, z down-screen (j/row maps to z), y up.
    fx = (i / (w - 1) - 0.5) * size if w > 1 else 0.0
    fz = (j / (h - 1) - 0.5) * size if h > 1 else 0.0
    return fx, fz

def _kind_from_label(label):
    # Resource kind from the material label under the resource's cell.
    # Keyword match (case-insensitive) on metal/energy/oil; anything else is
    # "neutral" (decorative / game decides). Documented rule, not a guess:
    # MAGER resources carry no kind of their own, so the material underneath
    # is the only signal available.
    lab = str(label or "").lower()
    for kind in ("metal", "energy", "oil"):
        if kind in lab:
            return kind
    return "neutral"

def build_map_spec():
    """Build a map-spec dict (spec/map-spec.schema.json) from the last Feature 2 generation.

    Returns (spec, artifacts) where artifacts is a list of
    (filename, ndarray, kind) with kind "u16" (heightmap) or "rgba" (splatmap).
    Referenced PNG filenames are relative to the spec JSON's folder.
    Raises gr.Error("Generate first") when GEN is empty."""
    g = GEN.get("grid")
    if g is None:
        raise gr.Error("Generate first.")
    g = np.asarray(g)
    if g.ndim != 2:
        raise gr.Error("Generate first.")
    h, w = int(g.shape[0]), int(g.shape[1])
    es = GEN.get("entries") or []
    try:
        seed = int(GEN.get("seed"))
    except (TypeError, ValueError):
        seed = 0

    import datetime
    name = "map_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    height_fname = "%s_height.png" % name
    splat_fname = "%s_splat.png" % name

    # Heightfield: explicit 16-bit PNG. GEN["height"] is pfield() output,
    # already min-max normalized to [0, 1] by construction.
    nz = GEN.get("height")
    try:
        nz = np.asarray(nz, dtype=np.float64)
    except Exception:
        raise gr.Error("Generation has no usable height field. Regenerate the map.")
    if nz.shape != (h, w):
        raise gr.Error("Height field shape %s != grid shape (%d, %d). Regenerate." % (nz.shape, h, w))
    # minHeight/maxHeight are normalized units: the PNG spans the data range
    # [0, 1] and the consumer (game profile) multiplies by its own height
    # scale. Honest because pfield() guarantees the [0, 1] normalization.
    height_u16 = (np.clip(nz, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    heightfield = {"mode": "explicit", "png": height_fname,
                   "minHeight": 0.0, "maxHeight": 1.0}
    if w >= 16:  # schema requires resolution >= 16 when present; MAGER allows 8px grids
        heightfield["resolution"] = w

    # Materials palette from the generation's entries.
    palette = []
    for pos, e in enumerate(es):
        try:
            eid = max(0, int(getattr(e, "id", pos)))
        except (TypeError, ValueError):
            eid = pos
        col = str(getattr(e, "color_hex", "") or "")
        if len(col) != 7 or not col.startswith("#"):
            col = "#808080"
        palette.append({"id": eid, "label": str(getattr(e, "label", "seg") or "seg"),
                        "color": col})

    # Splatmap: the generator's textured render as RGBA when available,
    # else per-material channel masks (same convention as the "c) RGBA
    # splatmap PNG" export).
    splat = None
    try:
        _t = np.asarray(GEN.get("texture"), dtype=np.uint8) if GEN.get("texture") is not None else None
    except Exception:
        _t = None
    if _t is not None and _t.ndim == 3 and _t.shape[2] == 3 and _t.shape[:2] == (h, w):
        splat = np.dstack([_t, np.full((h, w), 255, dtype=np.uint8)])
    else:
        k = min(4, max(1, len(es)))
        splat = np.zeros((h, w, 4), dtype=np.uint8)
        for i in range(k):
            splat[:, :, i] = np.where(g == i, 255, 0).astype(np.uint8)

    # Resources: GEN stores (base_id, cell_x, cell_y). Kind comes from the
    # material label under the resource's cell (see _kind_from_label).
    # Reserve default 2400 matches the rts deposit size; a game profile may
    # override per kind later.
    resources = []
    for _r in (GEN.get("resources") or []):
        try:
            _b, _xx, _yy = int(_r[0]), int(_r[1]), int(_r[2])
        except (TypeError, ValueError, IndexError):
            continue
        if not (0 <= _xx < w and 0 <= _yy < h):
            continue
        try:
            _lab = es[int(g[_yy, _xx])].label
        except (IndexError, ValueError, TypeError):
            _lab = ""
        _wx, _wz = _cell_to_world(_xx, _yy, w, h)
        resources.append({"kind": _kind_from_label(_lab),
                          "x": round(_wx, 2), "z": round(_wz, 2), "reserve": 2400})

    # Bases: centroid of each base's cells -> world coords. Pad radius is
    # 10 cells, matching the resource clustering radius in place_resources();
    # scenery clearRadius covers the same disc.
    cell = 600.0 / w if w else 600.0
    base_radius = round(10.0 * cell, 2)
    bases = []
    _bi = GEN.get("base_idx")
    if _bi is not None:
        try:
            _bi = np.asarray(_bi)
            for _b in sorted(int(v) for v in np.unique(_bi) if int(v) >= 0):
                _ys, _xs = np.where(_bi == _b)
                if len(_xs) == 0:
                    continue
                _wx, _wz = _cell_to_world(float(_xs.mean()), float(_ys.mean()), w, h)
                bases.append({"x": round(_wx, 2), "z": round(_wz, 2),
                              "radius": base_radius})
        except Exception as e:
            print("mager: base extraction failed (%s)" % e, file=sys.stderr)

    spec = {
        "meta": {"spec": "mager-map", "version": 1, "name": name, "seed": seed},
        "heightfield": heightfield,
        # 600x600 is the first consumer's (rts) world size; other games read
        # their own size from their profile or override these fields.
        "world": {"width": 600, "depth": 600},
        "materials": {"palette": palette, "splatmapPng": splat_fname},
        "resources": resources,
        "bases": bases,
        # No density control in the UI yet; 0.5 is the neutral default.
        "scenery": {"density": 0.5, "seed": seed, "clearRadius": base_radius},
        "profiles": {"rts": {"layer": "surface", "biome": "verdant"}},
    }
    artifacts = [(height_fname, height_u16, "u16"), (splat_fname, splat, "rgba")]
    return spec, artifacts

def f4_spec():
    """Export map spec: build from GEN, validate, then write JSON + PNGs.

    Validation happens BEFORE any write: on failure nothing is written."""
    spec, artifacts = build_map_spec()
    validate_spec(spec)
    d = get_export_dir()
    paths = []
    for fname, arr, kind in artifacts:
        p = os.path.join(d, fname)
        if kind == "u16":
            cv2.imwrite(p, arr)
        else:
            Image.fromarray(arr, "RGBA").save(p)
        paths.append(p)
    jp = os.path.join(d, spec["meta"]["name"] + ".map.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
    paths.append(jp)
    return "Map spec exported (schema-validated):\n" + "\n".join(_rel(p) for p in paths)

def _reveal_exports():
    p = os.path.abspath(get_export_dir())
    try:
        import subprocess
        if sys.platform.startswith("win"):
            os.startfile(p)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", p], check=True)
        else:
            subprocess.run(["xdg-open", p], check=True)
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
                js="(a) => { document.querySelectorAll(\"[id^='reallay']\").forEach(function(lay){"
                   " lay.style.opacity = (parseFloat(a) || 0).toString(); }); }")
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
        with gr.Tab("🎨 Feature 3 - Texture Swap"):
            gr.Markdown("Upload an existing map, pick target material(s) + replacement texture. "
                        "Mask-based OpenCV blend layers re-texture only matched pixels - geometry is preserved.")
            with gr.Row():
                sw_img = gr.Image(label="Existing map image", type="numpy")
                sw_json = gr.File(label="resource_label.json (or reuse Scan/Generate)", file_types=[".json"])
            sw_load = gr.Button("Load map + materials", variant="primary")
            sw_targets = gr.CheckboxGroup(label="Target materials to replace (masks)", choices=[], value=[])
            sw_repl = gr.Dropdown(label="Replacement texture (material exemplar)", choices=[], value=None)
            with gr.Row():
                sw_tol = gr.Slider(10, 250, value=90, step=1, label="Colour tolerance")
                sw_feather = gr.Slider(0, 8, value=1, step=1, label="Edge feather")
                sw_strength = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Blend strength")
                sw_tile = gr.Slider(0.25, 4.0, value=1.0, step=0.25, label="Texture scale")
            with gr.Row():
                sw_apply = gr.Button("Apply swap", variant="primary")
                sw_undo = gr.Button("Undo / Reset")
                sw_save = gr.Button("Save result")
            sw_out = gr.Image(label="Re-textured result")
            sw_msg = gr.Textbox(label="Swap log")
            sw_load.click(f3_upload, [sw_img, sw_json], [sw_out, sw_targets, sw_repl, sw_msg])
            sw_apply.click(f3_apply, [sw_targets, sw_repl, sw_tol, sw_feather, sw_strength, sw_tile], [sw_out, sw_msg])
            sw_undo.click(f3_undo, [], [sw_out, sw_msg])
            sw_save.click(f3_save, [], [sw_msg])
        with gr.Tab("📦 Feature 4 - Game Exporter"):
            gr.Markdown("Derives engine-ready files from the last Feature 2 generation.")
            ex_fmt = gr.CheckboxGroup(label="Export formats",
                choices=["a) 2D grid matrix JSON", "b) 16-bit heightmap PNG",
                         "c) RGBA splatmap PNG", "d) object/resource placement JSON"],
                value=["a) 2D grid matrix JSON"])
            ex_btn = gr.Button("Export selected", variant="primary")
            ex_msg = gr.Textbox(label="Export log")
            ex_btn.click(f4_export, [ex_fmt], [ex_msg])
            with gr.Row():
                sp_btn = gr.Button("🗺 Export map spec", variant="primary")
            sp_msg = gr.Textbox(label="Map spec export (schema-validated)")
            sp_btn.click(f4_spec, None, [sp_msg])
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

def _pick_port(first=GRADIO_PORT, last=GRADIO_PORT + 10):
    """First free TCP port in [first, last] on 127.0.0.1 (bind-test)."""
    import socket
    for port in range(first, last + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free port in %d..%d" % (first, last))

def launch():
    port = _pick_port()
    if port != GRADIO_PORT:
        print("mager: port %d busy, using %d" % (GRADIO_PORT, port))
    demo = build_ui()
    _srv_err = {}
    def _run():
        try:
            demo.launch(server_name="127.0.0.1", server_port=port,
                inbrowser=False, show_error=True, prevent_thread_lock=True, quiet=True)
        except Exception as e:
            _srv_err["exc"] = e
    threading.Thread(target=_run, daemon=True).start()
    url = "http://127.0.0.1:%d" % port
    import urllib.request
    for _ in range(60):
        if _srv_err.get("exc") is not None:
            break
        try:
            urllib.request.urlopen(url, timeout=2)
            break
        except Exception:
            time.sleep(0.5)
    if _srv_err.get("exc") is not None:
        raise RuntimeError("Gradio server failed to start on %s: %s" % (url, _srv_err["exc"]))
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
