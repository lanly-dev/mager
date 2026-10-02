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
    for r in data.get("resources", []):
        out.append(Res(int(r["id"]), str(r.get("label","seg")),
            str(r.get("color_hex","#808080")),
            str(r.get("mean_color_hex","#808080")),
            float(r.get("coverage",0)), float(r.get("threshold",0))))
    return out

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

def f1_scan(image, k, mode, morph, min_area, alpha):
    image = norm_img(image)
    if image is None:
        return _empty_view(), None, [], "Upload or paste an image first (upload / drag-drop / Ctrl+V)."
    if str(mode).lower().startswith("object"):
        vis, info = seg_objects(image, int(k), morph=int(morph), min_area=int(min_area))
    else:
        vis, info = seg_km(image, int(k), morph=int(morph))
    th = def_th(len(info))
    rows = []
    for i, s in enumerate(info):
        pb = PAL[s["id"] % len(PAL)]
        rows.append([s["id"], s["mean"], round(s["cov"]*100,2),
                     NAMES[i % len(NAMES)], bgr_hex((int(pb[0]),int(pb[1]),int(pb[2]))), th[i]])
    SCAN["info"] = info
    SCAN["vis"] = vis
    SCAN["orig"] = image
    view = _blend_view(_to_b64(vis), _to_b64(image), alpha)
    return view, rows, "Found %d segments (%s). Drag overlay slider - blends instantly." % (len(info), mode)

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
        if len(vals) < 6:
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
    fd, p = tempfile.mkstemp(prefix="resource_label_", suffix=".json")
    os.close(fd)
    open(p,"w",encoding="utf-8").write(json.dumps(pay, indent=2))
    try:
        hp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resource_label.json")
        open(hp,"w",encoding="utf-8").write(json.dumps(pay, indent=2))
    except Exception:
        pass
    return p
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

def domap(nz, es):
    order = sorted(range(len(es)), key=lambda k: es[k].threshold, reverse=True)
    g = np.full(nz.shape, order[-1], dtype=np.int32)
    for k in order:
        g[nz >= es[k].threshold] = k
    return g

def drawmap(g, es, ps):
    cv = np.zeros((g.shape[0], g.shape[1], 3), dtype=np.uint8)
    for k, e in enumerate(es):
        try:
            rgb = hex_rgb(e.color_hex)
        except Exception:
            rgb = (128,128,128)
        cv[g==k] = rgb
    im = Image.fromarray(cv, mode="RGB")
    if ps > 1:
        im = im.resize((g.shape[1]*ps, g.shape[0]*ps), Image.NEAREST)
    return im

def add_legend(im, es, g):
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
    return out

def f2_generate(jf, w, h, sc, oc, sd, ps):
    es = GEN.get("entries")
    if es is None:
        es, _ = load_entries(jf)
    w = max(8, min(int(w), 512))
    h = max(8, min(int(h), 512))
    nz = pfield(w, h, float(sc), int(oc), int(sd))
    g = domap(nz, es)
    GEN.update({"grid": g, "seed": int(sd)})
    im = add_legend(drawmap(g, es, int(ps)), es, g)
    cnt = np.bincount(g.ravel(), minlength=len(es))
    tot = cnt.sum()
    parts = ["%s:%.1f pct" % (es[k].label, 100.0*cnt[k]/tot) for k in range(len(es))]
    st = "Seed=%s %dx%d | " % (sd, w, h) + ", ".join(parts)
    return np.array(im), st

def f2_regen(jf, w, h, sc, oc, ps):
    return f2_generate(jf, w, h, sc, oc, random.randint(0, 999999), ps)

def f2_png():
    if GEN.get("grid") is None:
        raise gr.Error("Generate first.")
    es = GEN["entries"]
    im = add_legend(drawmap(GEN["grid"], es, 8), es, GEN["grid"])
    fd, p = tempfile.mkstemp(prefix="procedural_map_", suffix=".png")
    os.close(fd)
    im.save(p)
    return p

def f2_mat():
    if GEN.get("grid") is None:
        raise gr.Error("Generate first.")
    es = GEN["entries"]
    g = GEN["grid"]
    pay = {"meta": {"seed": GEN["seed"]}, "resources": [asdict(e) for e in es],
           "grid_ids": g.tolist(),
           "grid_labels": [[es[int(v)].label for v in row] for row in g.tolist()]}
    fd, p = tempfile.mkstemp(prefix="generated_map_matrix_", suffix=".json")
    os.close(fd)
    open(p,"w",encoding="utf-8").write(json.dumps(pay))
    return p

def build_ui():
    with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft()) as d:
        gr.Markdown("# " + APP_TITLE + " - offline Tab1 scan, Tab2 generate")
        with gr.Tab("Feature 1 - Scan"):
            with gr.Row():
                with gr.Column():
                    im = gr.Image(type="numpy", label="Upload or paste image (click to browse, drag-drop, or Ctrl+V)",
                                  sources=["upload", "clipboard"])
                    kk = gr.Slider(2, 12, value=5, step=1, label="Colors (K)")
                    md = gr.Radio(["Object lumps", "Color lumps"], value="Object lumps",
                        label="Segment mode")
                    mp = gr.Slider(3, 15, value=7, step=2, label="Cleanup size (bigger = more solid lumps)")
                    ma = gr.Slider(0, 5000, value=400, step=50, label="Min object area px (object mode)")
                    sb = gr.Button("Scan Image", variant="primary")
                with gr.Column():
                    so = gr.HTML(value=_empty_view(), label="Segmented (tint + real-map overlay)")
                    ov = gr.Slider(0.0, 1.0, value=0.35, step=0.01,
                        label="Real-map overlay alpha (0 = pure tint, 1 = real map)",
                        elem_id="ovslider")
                    st = gr.Textbox(label="Status", interactive=False)
            tb = gr.Dataframe(headers=["Segment ID","Mean Color","Pixel pct","Label","Display Color","Threshold"],
                datatype=["number","str","number","str","str","number"],
                col_count=(6,"fixed"), interactive=True, wrap=True, label="Segments")
            with gr.Row():
                eb = gr.Button("Export resource_label.json", variant="primary")
                jo = gr.File(label="resource_label.json")
            sb.click(f1_scan, [im, kk, md, mp, ma, ov], [so, tb, st])
            ov.input(None, [ov], None,
                js="(a) => { var lay = document.getElementById('reallay');"
                   " if (lay) lay.style.opacity = (parseFloat(a) || 0).toString(); }")
            eb.click(f1_export, [tb], [jo])
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
                    with gr.Row():
                        gb = gr.Button("Generate", variant="primary")
                        rb = gr.Button("Regenerate")
                with gr.Column():
                    mo = gr.Image(label="Map plus legend")
                    ss = gr.Textbox(label="Stats", interactive=False)
                    with gr.Row():
                        pb = gr.Button("Export PNG")
                        mb = gr.Button("Export grid JSON")
                    with gr.Row():
                        po = gr.File(label="PNG")
                        mo2 = gr.File(label="generated_map_matrix.json")
            lb.click(lambda f: load_entries(f)[1], [jf], [mi])
            gb.click(f2_generate, [jf, wi, hi, sc, oc, sd, ps], [mo, ss])
            rb.click(f2_regen, [jf, wi, hi, sc, oc, ps], [mo, ss])
            pb.click(lambda: f2_png(), None, [po])
            mb.click(lambda: f2_mat(), None, [mo2])
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
