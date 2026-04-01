import os, json, re, argparse, shutil
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import clip
from ultralytics import YOLO
from collections import defaultdict
from difflib import SequenceMatcher

from paddleocr import PaddleOCR

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def reset_output_dir(path):
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
        except Exception:
            pass
    os.makedirs(path, exist_ok=True)


ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    show_log=False
)

def extract_timestamp(frame):
    h, w = frame.shape[:2]
    crop = frame[0:int(h*0.12), 0:int(w*0.6)]
    
    result = ocr.ocr(crop)

    text = ""
    if result:
        for line in result:
            for box in line:
                text += " " + box[1][0]

    match = re.search(r"\d{2,4}[-/]\d{2}[-/]\d{2,4}\s+\d{2}:\d{2}:\d{2}", text)

    if match:
        return match.group(0)

    return None

def extract_location(frame):

    h, w = frame.shape[:2]

    crop = frame[int(h*0.88):h, int(w*0.35):w]

    result = ocr.ocr(crop)

    text = ""

    if result:
        for line in result:
            for box in line:
                text += " " + box[1][0]

    return text.strip()



def mmss(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))

def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]

def bbox_aspect(bb):
    x1, y1, x2, y2 = bb
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    return bw / bh

def aspect_sim(a, b, tol=1.2):
    if a is None or b is None:
        return 0.5
    d = abs(float(a) - float(b))
    return max(0.0, 1.0 - d / tol)

@torch.no_grad()
def clip_embed_bgr(bgr, model, preprocess, device):
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    img = preprocess(pil).unsqueeze(0).to(device)
    feat = model.encode_image(img)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu().numpy().astype(np.float32)

@torch.no_grad()
def clip_embed_text(text: str, model, device):
    tokens = clip.tokenize([text]).to(device)
    feat = model.encode_text(tokens)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu().numpy().astype(np.float32)

@torch.no_grad()
def clip_embed_image_path(img_path, model, preprocess, device):
    img = Image.open(img_path).convert("RGB")
    img = preprocess(img).unsqueeze(0).to(device)
    feat = model.encode_image(img)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu().numpy().astype(np.float32)

def safe_mean(vecs):
    if len(vecs) == 0:
        return None
    m = np.mean(np.stack(vecs, axis=0), axis=0)
    m = m / (np.linalg.norm(m) + 1e-12)
    return m.astype(np.float32)

def major_class(classes):
    if not classes:
        return None
    vals, cnts = np.unique(np.array(classes, dtype=np.int32), return_counts=True)
    return int(vals[np.argmax(cnts)])

def vehicle_color_label(bgr):
    if bgr is None or bgr.size == 0:
        return "unknown", 0.0
    h, w = bgr.shape[:2]
    if h == 0 or w == 0:
        return "unknown", 0.0

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    H = hsv[..., 0].astype(np.int32)   # 0..179
    S = hsv[..., 1].astype(np.int32)   # 0..255
    V = hsv[..., 2].astype(np.int32)   # 0..255
    L = lab[..., 0].astype(np.int32)

    valid = (V > 30) & (V < 245)
    valid_count = int(valid.sum())
    total = max(1, h * w)

    if valid_count > 50:
        s_med = float(np.median(S[valid]))
        v_med = float(np.median(V[valid]))
        l_med = float(np.median(L[valid]))
    else:
        s_med = float(np.median(S))
        v_med = float(np.median(V))
        l_med = float(np.median(L))

    bright_frac = float(((V >= 220) & valid).sum()) / total
    near_white_rgb = float(((S < 40) & (V > 200) & valid).sum()) / total
    low_sat_frac = float(((S < 50) & valid).sum()) / total

    if s_med < 40:
        if v_med < 70 or l_med < 55:
            conf = min(1.0, 0.6 + (70 - v_med) / 140.0)
            return "black", float(conf)
        if (v_med > 200 or l_med > 200) and (near_white_rgb > 0.02 or bright_frac > 0.05):
            conf = min(1.0, 0.55 + 0.45 * max((v_med - 200) / 55.0, bright_frac * 4.0))
            return "white", float(conf)
        conf = min(1.0, 0.4 + 0.6 * low_sat_frac)
        return "gray", float(conf)

    color_mask = valid & (S > 80) & (V > 40)
    mask_count = int(color_mask.sum())

    if mask_count < 40:
        if (v_med > 200 or l_med > 200) and low_sat_frac > 0.25:
            conf = min(1.0, 0.55 + 0.45 * (v_med - 200) / 55.0)
            return "white", float(conf)
        conf = min(1.0, 0.45 + 0.55 * (mask_count / 40.0))
        return "gray", float(conf)

    h_vals = H[color_mask].astype(np.uint8)
    hist = cv2.calcHist([h_vals], [0], None, [36], [0, 180]).flatten()
    dominant_bin = int(np.argmax(hist))
    bin_sum = float(hist.sum() + 1e-12)
    dom_value = float(hist[dominant_bin])
    h_dom = (dominant_bin + 0.5) * (180.0 / 36.0)

    base_conf = float(min(1.0, 0.35 + 0.65 * (dom_value / bin_sum)))
    sat_factor = min(1.0, s_med / 255.0 + 0.2)
    conf = float(min(1.0, base_conf * sat_factor + 0.05))

    if ((h_dom < 10) or (h_dom >= 170)) and (40 < s_med < 150) and (v_med > 170):
        return "pink", conf

    if (h_dom < 10) or (h_dom >= 170):
        return "red", conf
    if 10 <= h_dom < 25:
        return "orange", conf
    if 25 <= h_dom < 35:
        return "yellow", conf
    if 35 <= h_dom < 85:
        return "green", conf
    if 85 <= h_dom < 135:
        return "blue", conf
    if 135 <= h_dom < 170:
        return "purple", conf

    return "unknown", conf

def hsv_hist(bgr, bins=(16, 16, 16)):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h_bins, s_bins, v_bins = bins
    hist = cv2.calcHist([hsv], [0, 1, 2], None,
                        [h_bins, s_bins, v_bins],
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
    return hist

def compute_stopped_time(centers, fps, bboxes=None, min_run_steps=5, sample_every=1):
    if len(centers) < 2:
        return 0.0
    
    c = np.array(centers, dtype=np.float32)
    d = np.linalg.norm(c[1:] - c[:-1], axis=1)
    
    move_thresh = 3.0
    if bboxes is not None and len(bboxes) > 0:
        avg_w = np.mean([b[2] - b[0] for b in bboxes])
        avg_h = np.mean([b[3] - b[1] for b in bboxes])
        diag = np.sqrt(avg_w**2 + avg_h**2)
        move_thresh = max(1.5, diag * 0.02) 

    still = (d < float(move_thresh)).astype(np.int32)
    total_steps = 0
    run = 0
    for v in still:
        if v == 1:
            run += 1
        else:
            if run >= int(min_run_steps):
                total_steps += run
            run = 0
    if run >= int(min_run_steps):
        total_steps += run
    step_sec = max(1e-6, float(sample_every) / float(fps))
    return float(total_steps * step_sec)

def draw_track_frame(video_path, frame_id, bbox, out_path, label):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(out_path, frame)
    return True

VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck

def get_coco_names(yolo: YOLO):
    try:
        names = yolo.model.names
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, (list, tuple)):
            return {i: str(n) for i, n in enumerate(names)}
    except Exception:
        pass
    return {i: f"class_{i}" for i in range(80)}

# ========================= query parsing
COLOR_WORDS = {"red", "blue", "white", "black", "silver", "gray", "grey", "yellow", "green", "orange", "purple", "pink"}
SIMILAR_COLORS = {
    "white": ["white", "gray", "silver"],
    "gray": ["gray", "silver", "white"],
    "grey": ["gray", "silver", "white"],
    "silver": ["silver", "gray", "white"],
}
VEHICLE_WORDS = {
    "car": 2,
    "truck": 7,
    "bus": 5,
    "motorcycle": 3,
    "bike": 3,
    "auto": 2,
    "autorickshaw": 2,
    "rickshaw": 2,
    "person": 0,
    "bicycle": 1,
}

LOCATION_CAMERA_MAP = {
    "subashchandrabose": "cam1",
    "kathrikkadavu": "cam2",
    "petta": "cam3",
}

STOP_WORDS = {"jn", "road", "street", "junction"}

def _mmss_to_sec(ts: str) -> int:
    parts = ts.split(":")
    if len(parts) == 2:
        m, s = map(int, parts)
        return m * 60 + s
    elif len(parts) == 3:
        h, m, s = map(int, parts)
        return h * 3600 + m * 60 + s
    return 0

def parse_text_query(q: str, blast_time: str = None):
    qq = q.strip()
    qlow = qq.lower()
    out = {
        "raw": qq,
        "cameras": [],
        "locations": [],
        "time_range": None,
        "after_sec": None,
        "before_sec": None,
        "colors": [],
        "dom_cls": None,
        "stop_min_sec": None,
        "near_blast": None,
        "free_text": qq,
        "location_keywords": [],
    }

    if is_plate_query(qq) if 'is_plate_query' in globals() else False:
        out["is_plate_query"] = True
        out["plate_query"] = normalize_plate(qq)
        out["free_text"] = qq
        out["location_keywords"] = []
        return out
    cams = set()
    for m in re.findall(r"\bcam\s*([0-9]+)\b", qlow):
        cams.add(f"cam{int(m)}")
    for m in re.findall(r"\bcamera[-\s]*([0-9]+)\b", qlow):
        cams.add(f"cam{int(m)}")
    if cams:
        out["cameras"] = sorted(cams)
    m = re.search(r"\bbetween\s+(\d{1,2}:\d{2})\s*(?:-|–|to)\s*(\d{1,2}:\d{2})\b", qlow)
    if m:
        out["time_range"] = (_mmss_to_sec(m.group(1)), _mmss_to_sec(m.group(2)))
    m = re.search(r"\bafter\s+(\d{1,2}:\d{2}(?::\d{2})?)", qlow)
    if m:
        out["after_sec"] = _mmss_to_sec(m.group(1))

    m = re.search(r"\bbefore\s+(\d{1,2}:\d{2}(?::\d{2})?)", qlow)
    if m:
        out["before_sec"] = _mmss_to_sec(m.group(1))
    if "blast" in qlow:
        wm = re.search(r"\bwithin\s+(\d+)\s*(min|mins|minutes|sec|secs|seconds)\b", qlow)
        window_sec = 10
        if wm:
            val = int(wm.group(1))
            unit = wm.group(2)
            window_sec = val if "sec" in unit else val * 60
        if blast_time:
            blast_sec = _mmss_to_sec(blast_time)
            out["near_blast"] = {"blast_time": blast_time, "blast_sec": blast_sec,
                                 "window_before": window_sec, "window_after": window_sec}
    for c in COLOR_WORDS:
        if re.search(rf"\b{re.escape(c)}\b", qlow):
            out["colors"].append("gray" if c == "grey" else c)
    m = re.search(r"\b(stop|stopping|stopped)\b.*\b(longer than|over|>=)\s*(\d+(?:\.\d+)?)\s*(min|mins|minutes|sec|secs|seconds)\b", qlow)
    if m:
        x = float(m.group(3))
        unit = m.group(4)
        out["stop_min_sec"] = float(x if "sec" in unit else x * 60)
    for word, cid in VEHICLE_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", qlow):
            if word == "auto":
                out["dom_cls"] = None
            else:
                out["dom_cls"] = int(cid)
            out["dom_cls_name"] = word
            break
    free = qq
    for c in out["colors"]:
        free = re.sub(rf"\b{re.escape(c)}\b", "", free, flags=re.I)
    free = re.sub(r"\bcam\s*\d+\b", "", free, flags=re.I)
    free = re.sub(r"\bcamera[-\s]*\d+\b", "", free, flags=re.I)
    free = re.sub(r"\bbetween\s+\d{1,2}:\d{2}\s*(?:-|–|to)\s*\d{1,2}:\d{2}\b", "", free, flags=re.I)
    free = re.sub(r"\b(after|before)\s+\d{1,2}:\d{2}(?::\d{2})?", "", free, flags=re.I)
    free = re.sub(r"\b(in|at|on|between|after|before|to)\b", "", free, flags=re.I)
    out["free_text"] = " ".join(free.split()).strip() or qq
    STOPWORDS = {
    "car","bus","truck","vehicle","bike","motorcycle",
    "stopped","stop","stopping",
    "longer","than","seconds","second","sec","secs",
    "after","before","between","over","greater",
    "cam1","cam2","cam3",
    "in","at","on","near"
    }

    words = re.findall(r"[a-zA-Z]+", qlow)

    out["location_keywords"] = [
        w for w in words
        if w not in STOPWORDS
        and w not in COLOR_WORDS
        and w not in VEHICLE_WORDS
    ]
    for kw in out["location_keywords"]:
        if kw.lower() in LOCATION_CAMERA_MAP:
            out["cameras"] = [LOCATION_CAMERA_MAP[kw.lower()]]
    return out

# ========================= filtering
def filter_tracks(tracks, constraints, cam_id_to_name):
    if constraints is None:
        return tracks
    cams = constraints.get("cameras") or []
    colors = [c.lower() for c in (constraints.get("colors") or [])]
    dom_cls = constraints.get("dom_cls", None)
    stop_min = constraints.get("stop_min_sec", None)
    time_range = constraints.get("time_range", None)
    after_sec = constraints.get("after_sec", None)
    before_sec = constraints.get("before_sec", None)
    near_blast = constraints.get("near_blast", None)
    location_keywords = constraints.get("location_keywords") or []
    
    out = []
    for t in tracks:
        cam_name = t.get("cam_name") or cam_id_to_name.get(int(t["cam"]), f"cam{int(t['cam'])+1}")
        if cams and cam_name not in cams:
            continue
        
        location_match = False

        if location_keywords:
            t_loc = re.sub(r'[^a-zA-Z]', '', (t.get("location") or "")).lower()

            for kw in location_keywords:
                kw_clean = re.sub(r'[^a-zA-Z]', '', kw).lower()

                if kw_clean in t_loc:
                    location_match = True
                    break
            if not location_match:
                continue
                
        if dom_cls is not None:
            t_cls = t.get("dom_cls")

            if t_cls == dom_cls:
                pass
            else:
                if not (dom_cls in [5, 7] and t_cls in [5, 7]):
                    continue
                
        if colors:
            t_color = (t.get("color_label") or "").lower()
            matched = False
            for q_color in colors:
                if q_color in SIMILAR_COLORS:
                    if t_color in SIMILAR_COLORS[q_color] or t_color == q_color:
                        matched = True
                        break
                elif t_color == q_color:
                    matched = True
                    break
            if not matched:
                continue
                
        if stop_min is not None and float(t.get("stopped_time", 0.0)) < float(stop_min):
            continue
            
        ts = t.get("t_start_sec")
        te = t.get("t_end_sec")
        
        if ts is None or te is None:
            if time_range or after_sec or before_sec or near_blast:
                continue
            out.append(t)
            continue
            
        if time_range is not None:
            s, e = time_range
            if te < s or ts > e:
                continue
        if after_sec is not None and te < after_sec:
            continue
        if before_sec is not None and ts > before_sec:
            continue
        if near_blast and near_blast.get("blast_sec") is not None:
            bs = near_blast["blast_sec"]
            w1 = near_blast["window_before"]
            w2 = near_blast["window_after"]
            if te < (bs - w1) or ts > (bs + w2):
                continue
        out.append(t)
    return out

def make_html(results, out_dir):
    html_path = os.path.join(out_dir, "results.html")
    q = results["query"]
    cam_keys = sorted(results["matches_by_cam"].keys(), key=lambda x: int(re.search(r"\d+", x).group()))
    constraints = results.get("constraints", None)
    constraints_pre = ""
    if constraints:
        keys = ["cameras", "location_keywords", "time_range", "after_sec", "before_sec", "colors", "dom_cls",
                "stop_min_sec", "near_blast", "free_text"]
        lines = []
        for k in keys:
            if k in constraints and constraints[k] not in (None, [], "", {}):
                lines.append(f"{k}: {constraints[k]}")
        constraints_pre = "\n".join(lines) if lines else "(none)"
    query_text = q.get("text", None)
    query_type = q.get("type", "text")
    parts = [f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Cross-Camera Results</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 18px; color:#111; background:#f4f6f8; }}
h1 {{ margin:0 0 8px; }}
h2 {{ margin-top:18px; }}
.muted {{ color:#555; }}
.section {{ margin:14px 0; }}
.block {{ border:1px solid #e2e2e2; border-radius:10px; padding:12px 14px; background:#fff; }}
.pre {{ white-space:pre-line; font-family:monospace; font-size:13px; line-height:1.45; }}
.grid {{ display:grid; grid-template-columns:repeat(3, 1fr); gap:14px; }}
.card{{ border:1px solid #e2e2e2; border-radius:10px; padding:10px; min-height:320px; }}
img{{ width:100%; max-height:350px; object-fit:contain; border-radius:8px; }}
.cap {{ margin:0 0 8px; font-size:14px; white-space:pre-line; }}
.nf {{ color:#b00; font-weight:bold; }}
.small {{ font-size:13px; }}
.group-card {{ margin-bottom:12px; padding:10px; border-radius:8px; background:#fafafa; border:1px solid #eee; }}
</style>
</head>
<body>
  <h1>Cross-Camera Results</h1>
  <div class="section">
    <h2>Query</h2>
    <div class="grid">
      <div class="card">
        <div class="cap"><b>{'Image query' if query_type=='image' else 'Text query'}</b></div>
        <div class="block pre">{query_text or ""}</div>
      </div>
      <div class="card">
        <div class="cap"><b>Parsed constraints</b></div>
        <div class="block pre">{constraints_pre}</div>
      </div>
    </div>
  </div>
"""]
    groups = results.get("groups", None)
    if groups:
        parts.append("""<div class="section"><h2>Vehicle Groups (cross-camera)</h2>""")
        cams_all = cam_keys
        for gi, grp in enumerate(groups, start=1):
            parts.append(f"""<div class="group-card"><b>Search Results</b><div class="grid">""")
            grp_by_cam = {x["cam_name"]: x for x in grp}
            for cam in cams_all:
                item = grp_by_cam.get(cam)
                if item is None:
                    parts.append(f"""<div class="card"><div class="cap nf">{cam}: NOT FOUND</div></div>""")
                else:
                    real_time = item.get("timestamp")
                    if real_time:
                        ts = real_time
                    else:
                        ts = item.get("t_start","00:00")
                    image = item.get("image")
                    dom = item.get("dom_cls_name", "")
                    stopped = item.get("stopped_time")
                    location = item.get("location")

                    extra = []

                    if stopped and stopped > 0:
                        extra.append(f"stopped: {stopped:.2f}s")

                    if location:
                        extra.append(location)

                    extra_text = " | ".join(extra)

                    plate = item.get("plate")
                    plate_text = f" | {plate}" if plate else ""

                    caption = f"<b>{cam}</b> — {ts} | {dom}{plate_text}"
                    if extra_text:
                        caption += " | " + extra_text

                    parts.append(
                        f"""<div class="card">
                        <div class="cap small">{caption}</div>
                        <img src="{image}" />
                        </div>"""
                    )
            parts.append("</div></div>")
    parts.append("</body></html>")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return html_path
def detect_plate_and_ocr(rep_crop, plate_yolo=None):

    if rep_crop is None or rep_crop.size == 0:
        return None, 0.0

    plate_img = None
    best_conf = 0.0

    # ===================== DEBUG SAVE INPUT =====================
    cv2.imwrite("debug_plate_input.png", rep_crop)

    # ===================== 1. PLATE DETECTION =====================
    try:
        if plate_yolo is not None:
            res = plate_yolo.predict(
                rep_crop,
                conf=0.25,
                iou=0.5,
                imgsz=640,
                device=0 if torch.cuda.is_available() else "cpu",
                verbose=False
            )

            if res and len(res) > 0:
                r = res[0]

                print("Plate detection boxes:",
                      len(r.boxes) if getattr(r, "boxes", None) is not None else 0)

                if getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
                    best_box = None

                    for b in r.boxes:
                        conf = float(b.conf.item())

                        if conf > best_conf and conf > 0.4:
                            best_conf = conf
                            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()

                            bb = clamp_box(
                                x1, y1, x2, y2,
                                rep_crop.shape[1],
                                rep_crop.shape[0]
                            )
                            best_box = bb

                    if best_box is not None:
                        x1, y1, x2, y2 = best_box

                        pad_x = int(0.15 * (x2 - x1))
                        pad_y = int(0.25 * (y2 - y1))

                        x1 = max(0, x1 - pad_x)
                        y1 = max(0, y1 - pad_y)
                        x2 = min(rep_crop.shape[1], x2 + pad_x)
                        y2 = min(rep_crop.shape[0], y2 + pad_y)

                        plate_img = rep_crop[y1:y2, x1:x2]

    except Exception as e:
        print("Plate detection error:", e)

    if plate_img is None or best_conf < 0.4:
        h, w = rep_crop.shape[:2]
        plate_img = rep_crop[int(h*0.5):int(h*0.9), int(w*0.1):int(w*0.9)]

    if plate_img is None or plate_img.size == 0:
        return None, 0.0

    try:
        plate_img = cv2.resize(plate_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        plate_img = cv2.bilateralFilter(plate_img, 11, 17, 17)

        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        gray = cv2.filter2D(gray, -1, kernel)

        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

    except Exception:
        thresh = plate_img

    # ===================== 4. OCR =====================
    candidates = []

    for img in [plate_img, gray, thresh]:
        try:
            result = ocr.ocr(img)

            if result:
                for line in result:
                    for box in line:
                        text = box[1][0]
                        conf = float(box[1][1])

                        clean = re.sub(r'[^A-Z0-9]', '', text.upper())

                        def looks_like_plate(text):
                            has_letter = any(c.isalpha() for c in text)
                            has_digit = any(c.isdigit() for c in text)

                            if not has_digit:
                                return False

                            if len(text) < 4:
                                return False

                            if has_letter and has_digit:
                                return True

                            if text.isdigit() and len(text) >= 5:
                                return True

                            return False

                        if looks_like_plate(clean):
                            candidates.append((clean, conf))

        except Exception:
            continue

    if not candidates:
        print("OCR FAILED")
        return None, 0.0

    cleaned = []
    for text, conf in candidates:
        cp = clean_plate(text)
        cleaned.append((cp, conf, is_valid_plate(cp)))

    valid_items = [x for x in cleaned if x[2]]
    best_text = None
    best_score = 0.0
    if valid_items:
        counts = {}
        for p, c, v in valid_items:
            counts.setdefault(p, []).append(c)
        best_plate = max(counts.items(), key=lambda kv: (len(kv[1]), max(kv[1])))[0]
        best_conf = max(counts[best_plate])
        best_text = best_plate
        best_score = min(1.0, 0.6 + best_conf)
    else:
        if cleaned:
            best_plate, best_conf, _ = max(cleaned, key=lambda x: x[1])
            best_text = best_plate
            best_score = min(1.0, 0.3 + best_conf)

    print("FINAL PLATE:", best_text)

    if not best_text:
        return None, 0.0
    return best_text, float(best_score)

def build_track_index(
    videos,
    out_dir="track_index",
    yolo_model="yolov8s.pt",
    clip_name="ViT-B/32",
    conf=0.25,
    iou=0.6,
    imgsz=640,
    sample_every=1,
    min_track_len=6,
    max_samples_per_track=25,
    tracker="bytetrack.yaml",
    plate_model_path=None,
):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load(clip_name, device=device)
    clip_model.eval()
    yolo = YOLO(yolo_model)
    plate_yolo = None
    if plate_model_path and os.path.exists(plate_model_path):
        try:
            plate_yolo = YOLO(plate_model_path)
            print("Loaded plate YOLO model:", plate_model_path)
        except Exception as e:
            print("Plate YOLO failed to load:", e)
            plate_yolo = None
    coco_names = get_coco_names(yolo)
    all_tracks = []
    for cam_id, vpath in enumerate(videos):
        cam_name = f"cam{cam_id + 1}"
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {vpath}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        stream = yolo.track(
            source=vpath,
            stream=True,
            persist=True,
            verbose=False,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            tracker=tracker
        )
        tracks = {}
        frame_idx = -1
        for r in tqdm(stream, total=total, desc=f"{cam_name} index", unit="frame"):
            frame_idx += 1
            if sample_every is not None and (frame_idx % sample_every != 0):
                continue
            if r.boxes is None or len(r.boxes) == 0:
                continue
            frame = r.orig_img
            h, w = frame.shape[:2]
            boxes = r.boxes
            tids = boxes.id
            if tids is None:
                continue
            for i in range(len(boxes)):
                cls = int(boxes.cls[i].item())
                if cls not in VEHICLE_CLASSES:
                    continue
                score = float(boxes.conf[i].item())
                if score < conf:
                    continue
                tid = int(tids[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()
                bb = clamp_box(x1, y1, x2, y2, w, h)
                if bb is None:
                    continue
                if tid not in tracks:
                    tracks[tid] = {
                        "cam": cam_id,
                        "cam_name": cam_name,
                        "video": os.path.basename(vpath),
                        "fps": float(fps),
                        "track_id": tid,
                        "frames": [],
                        "boxes": [],
                        "centers": [],
                        "classes": [],
                        "clip_vecs": [],
                        "hist_vecs": [],
                    }
                t = tracks[tid]
                t["frames"].append(int(frame_idx))
                t["boxes"].append(bb)
                t["classes"].append(cls)
                cx = 0.5 * (bb[0] + bb[2])
                cy = 0.5 * (bb[1] + bb[3])
                t["centers"].append((float(cx), float(cy)))
                if len(t["clip_vecs"]) < max_samples_per_track:
                    crop = frame[bb[1]:bb[3], bb[0]:bb[2]]
                    if crop.size > 0:
                        t["clip_vecs"].append(clip_embed_bgr(crop, clip_model, preprocess, device))
                        t["hist_vecs"].append(hsv_hist(crop))
        for tid, tr in tracks.items():
            if len(tr["frames"]) < min_track_len:
                continue
            vecs = np.stack(tr["clip_vecs"], axis=0)
            clip_emb = np.mean(vecs, axis=0)
            clip_emb = clip_emb / (np.linalg.norm(clip_emb) + 1e-12)
            clip_emb = clip_emb.astype(np.float32)
            if clip_emb is None:
                continue
            if len(tr["hist_vecs"]) == 0:
                continue
            hmean = np.mean(np.stack(tr["hist_vecs"], axis=0), axis=0).astype(np.float32)
            hmean = hmean / (np.linalg.norm(hmean) + 1e-12)
            cls_dom = major_class(tr["classes"])
            cls_name = coco_names.get(int(cls_dom), str(cls_dom)) if cls_dom is not None else "unknown"
            f_start, f_end = int(min(tr["frames"])), int(max(tr["frames"]))
            rep_idx = len(tr["frames"]) // 2
            rep_frame = int(tr["frames"][rep_idx])
            rep_box = tr["boxes"][rep_idx]
            rep_aspect = float(bbox_aspect(rep_box))
            t_start_sec = None
            t_end_sec = None
            try:
                cap2 = cv2.VideoCapture(vpath)
                cap2.set(cv2.CAP_PROP_POS_FRAMES, int(f_start))
                _ = cap2.read()
                ms = cap2.get(cv2.CAP_PROP_POS_MSEC)
                if ms and ms > 0:
                    t_start_sec = float(ms / 1000.0)
                cap2.release()
            except Exception:
                t_start_sec = None
            try:
                cap2 = cv2.VideoCapture(vpath)
                cap2.set(cv2.CAP_PROP_POS_FRAMES, int(f_end))
                _ = cap2.read()
                ms = cap2.get(cv2.CAP_PROP_POS_MSEC)
                if ms and ms > 0:
                    t_end_sec = float(ms / 1000.0)
                cap2.release()
            except Exception:
                t_end_sec = None
            if t_start_sec is None:
                t_start_sec = float(f_start / tr["fps"])
            if t_end_sec is None:
                t_end_sec = float(f_end / tr["fps"])
            dwell_time = float((f_end - f_start) / tr["fps"])
            stopped_time = compute_stopped_time(
                tr["centers"], tr["fps"],
                bboxes=tr["boxes"],
                min_run_steps=5,
                sample_every=sample_every
            )
            rep_crop_color = None
            rep_crop_for_plate = None
            try:
                cap2 = cv2.VideoCapture(vpath)
                cap2.set(cv2.CAP_PROP_POS_FRAMES, int(rep_frame))
                ok2, fr2 = cap2.read()
                cap2.release()
                timestamp = None
                location = None
                if ok2:
                    try:
                        timestamp = extract_timestamp(fr2)
                    except Exception:
                        timestamp = None
                    try:
                        location = extract_location(fr2)
                    except Exception:
                        location = None
                    x1, y1, x2, y2 = rep_box
                    rep_crop_color = fr2[y1:y2, x1:x2]
                    if rep_crop_color is not None:
                        h, w = rep_crop_color.shape[:2]

                        rep_crop_for_plate = rep_crop_color.copy()
                    else:
                        rep_crop_for_plate = None
            except Exception:
                rep_crop_color = None
                rep_crop_for_plate = None
                timestamp = None
                location = None
            color_label = "unknown"
            color_conf = 0.0
            if rep_crop_color is not None and rep_crop_color.size > 0:
                label, conf = vehicle_color_label(rep_crop_color)
                color_label = label
                color_conf = float(conf)
            plate_text = None
            plate_conf = 0.0

            sample_frames = tr["frames"][::max(1, len(tr["frames"]) // 5)]

            plate_candidates = []

            for f in sample_frames:
                try:
                    cap_tmp = cv2.VideoCapture(vpath)
                    cap_tmp.set(cv2.CAP_PROP_POS_FRAMES, int(f))
                    ok_tmp, frame_tmp = cap_tmp.read()
                    cap_tmp.release()

                    if not ok_tmp:
                        continue

                    # get corresponding bbox
                    idx = tr["frames"].index(f)
                    box = tr["boxes"][idx]

                    x1, y1, x2, y2 = box
                    crop = frame_tmp[y1:y2, x1:x2]

                    if crop is None or crop.size == 0:
                        continue

                    # lower plate region crop
                    h, w = crop.shape[:2]
                    crop_plate = crop[int(h*0.55):int(h*0.95), int(w*0.15):int(w*0.85)]

                    txt, conf = detect_plate_and_ocr(crop_plate, plate_yolo=plate_yolo)

                    if txt:
                        plate_candidates.append((txt, conf))

                except Exception:
                    continue

            plate_text = None
            plate_conf = 0.0
            if plate_candidates:
                cleaned = [(clean_plate(p), c, is_valid_plate(clean_plate(p))) for p, c in plate_candidates]
                valid = [x for x in cleaned if x[2]]
                if valid:
                    counts = {}
                    for p, c, v in valid:
                        counts.setdefault(p, []).append(c)
                    chosen = max(counts.items(), key=lambda kv: (len(kv[1]), max(kv[1])))[0]
                    plate_text = chosen
                    plate_conf = float(max(counts[chosen]))
                else:
                    plate_text, plate_conf, _ = max(cleaned, key=lambda x: x[1])
            all_tracks.append({
                "cam": tr["cam"],
                "cam_name": tr["cam_name"],
                "video": tr["video"],
                "fps": tr["fps"],
                "track_id": tr["track_id"],
                "timestamp": timestamp,
                "location": location,
                "dom_cls": int(cls_dom) if cls_dom is not None else None,
                "dom_cls_name": cls_name,
                "color_label": color_label,
                "color_conf": color_conf,
                "f_start": f_start,
                "f_end": f_end,
                "t_start": mmss(t_start_sec),
                "t_end": mmss(t_end_sec),
                "t_start_sec": float(t_start_sec),
                "t_end_sec": float(t_end_sec),
                "dwell_time": dwell_time,
                "stopped_time": float(stopped_time),
                "rep_frame": rep_frame,
                "rep_box": rep_box,
                "rep_aspect": rep_aspect,
                "clip": clip_emb.tolist(),
                "hist": hmean.tolist(),
                "plate": plate_text,
                "plate_conf": float(plate_conf),
            })
    path = os.path.join(out_dir, "tracks.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_tracks, f)
    print(f"Track index saved: {path} (tracks={len(all_tracks)})")
    return path

def load_track_index(path):
    with open(path, "r", encoding="utf-8") as f:
        tracks = json.load(f)
    for t in tracks:
        t["clip"] = np.array(t["clip"], np.float32)
        t["hist"] = np.array(t["hist"], np.float32)
    return tracks

def embed_query_with_yolo_largest(query_img_path, yolo, clip_model, preprocess, device, conf=0.25):
    bgr = cv2.imread(query_img_path)
    if bgr is None:
        raise RuntimeError(f"Cannot read {query_img_path}")
    res = yolo.predict(bgr, verbose=False)[0]
    best = None
    if res.boxes is not None:
        H, W = bgr.shape[:2]
        for b in res.boxes:
            cls = int(b.cls.item())
            score = float(b.conf.item())
            if score < conf:
                continue
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
            bb = clamp_box(x1, y1, x2, y2, W, H)
            if bb is None:
                continue
            area = (bb[2] - bb[0]) * (bb[3] - bb[1])
            if best is None or area > best[0]:
                best = (area, cls, bb)
    if best is None:
        q_clip = clip_embed_bgr(bgr, clip_model, preprocess, device)
        q_hist = hsv_hist(bgr)
        return q_clip, q_hist, None
    _, _, bb = best
    x1, y1, x2, y2 = bb
    crop = bgr[y1:y2, x1:x2]
    q_clip = clip_embed_bgr(crop, clip_model, preprocess, device)
    q_hist = hsv_hist(crop)
    q_aspect = bbox_aspect(bb)
    return q_clip, q_hist, q_aspect

def track_similarity(t1, t2, weights=None):
    if weights is None:
        weights = {'clip':0.55, 'hist':0.2, 'aspect':0.02, 'color':0.15, 'plate':0.08}
    try:
        if t1.get("timestamp") and t2.get("timestamp"):
            time1 = re.search(r"\d{2}:\d{2}:\d{2}", t1["timestamp"]).group()
            time2 = re.search(r"\d{2}:\d{2}:\d{2}", t2["timestamp"]).group()
            h1,m1,s1 = map(int,time1.split(":"))
            h2,m2,s2 = map(int,time2.split(":"))
            sec1 = h1*3600+m1*60+s1
            sec2 = h2*3600+m2*60+s2
            if abs(sec1-sec2) > 600:
                return 0.0
    except:
        pass

    c1 = t1.get("dom_cls")
    c2 = t2.get("dom_cls")

    if c1 is not None and c2 is not None:
        cls_score = 1.0 if c1 == c2 else 0.5
    
    asp1 = t1.get("rep_aspect")
    asp2 = t2.get("rep_aspect")

    if asp1 is not None and asp2 is not None:
        if abs(asp1 - asp2) > 1.4:
            return 0.0

    cs = cosine_sim(np.array(t1['clip'], np.float32), np.array(t2['clip'], np.float32))
    hs = cosine_sim(np.array(t1['hist'], np.float32), np.array(t2['hist'], np.float32))
    asp = aspect_sim(t1.get('rep_aspect', None), t2.get('rep_aspect', None), tol=1.2)

    col1 = (t1.get("color_label") or "").lower()
    col2 = (t2.get("color_label") or "").lower()

    color_score = 0.5  # Neutral default
    if col1 and col2:
        if col1 == "unknown" or col2 == "unknown":
            color_score = 0.5
        elif col1 == col2:
            color_score = 1.0
        elif col1 in ["gray", "silver"] and col2 in ["gray", "silver"]:
            color_score = 0.9
        elif col1 in ["white", "silver", "gray"] and col2 in ["white", "silver", "gray"]:
            color_score = 0.7
        else:
            color_Score =  0.2

    plate1 = normalize_plate(t1.get('plate')) if t1.get('plate') else None
    plate2 = normalize_plate(t2.get('plate')) if t2.get('plate') else None
    plate_score = 0.0
    conf1 = float(t1.get('plate_conf', 0.0) or 0.0)
    conf2 = float(t2.get('plate_conf', 0.0) or 0.0)
    # Only use plate when both reads have reasonable confidence
    if plate1 and plate2 and conf1 >= 0.7 and conf2 >= 0.7:
        sim = SequenceMatcher(None, plate1, plate2).ratio()
        if sim > 0.95:
            plate_score = 1.0
        elif sim > 0.80:
            plate_score = 0.85
        elif sim > 0.60:
            plate_score = 0.65

    total = 0.0
    total += weights.get('clip', 0.0) * max(0.0, cs)
    total += weights.get('hist', 0.0) * max(0.0, hs)
    total += weights.get('aspect', 0.0) * max(0.0, asp)
    total += weights.get('color', 0.0) * max(0.0, color_score)
    total += 0.1 * cls_score
    plate_weight = weights.get('plate', 0.0)
    if plate_score > 0.0:
        total += plate_weight * plate_score   
    return float(total)

def cross_camera_group_tracks(tracks, group_thresh=0.48, weights=None):
    n = len(tracks)
    if n == 0:
        return []
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if int(tracks[i]['cam']) == int(tracks[j]['cam']):
                continue
            sim = track_similarity(tracks[i], tracks[j], weights=weights)
            if sim >= group_thresh:
                adj[i].append(j)
                adj[j].append(i)
    visited = [False]*n
    groups = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        comp = []
        visited[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        if comp:
            grp = [tracks[idx].copy() for idx in comp]
            groups.append(grp)
    groups.sort(key=lambda g: -len(g))
    return groups

def group_results_by_rank(matches_by_cam):
    cams = sorted(matches_by_cam.keys(), key=lambda x: int(x.replace("cam","")))
    max_rank = 0
    for c in cams:
        if matches_by_cam[c] and matches_by_cam[c][0].get("status") != "NOT_FOUND":
            max_rank = max(max_rank, len(matches_by_cam[c]))
    grouped = []
    for r in range(max_rank):
        row = {"vehicle_id": r+1, "cams": {}}
        for cam in cams:
            items = matches_by_cam.get(cam, [])
            if len(items) > r and items[0].get("status") != "NOT_FOUND":
                row["cams"][cam] = items[r]
            else:
                row["cams"][cam] = None
        grouped.append(row)
    return grouped

def normalize_plate(p):
    return re.sub(r'[^A-Z0-9]', '', (p or "").upper())

def clean_plate(text: str) -> str:
    if not text:
        return ""
    text = text.upper()
    replacements = {
        'O': '0',
        'I': '1',
        'L': '1',
        'Z': '2',
        'S': '5',
        'B': '8',
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return re.sub(r'[^A-Z0-9]', '', text)

def is_valid_plate(text: str) -> bool:
    if not text:
        return False
    t = normalize_plate(text)
    pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$'
    return re.match(pattern, t) is not None

def is_plate_query(q: str) -> bool:
    if not q:
        return False
    return is_valid_plate(q.strip().upper())

def run_search(
    videos,
    index_path,
    query_text=None,
    blast_time=None,
    out_dir="results_out",
    clip_name="ViT-B/32",
    topk_per_cam=10,
    final_thresh=0.45,
    group_thresh=0.48,
    group_weights=None,
    plate_model_path=None,
):
    if query_text is None:
        raise ValueError("Provide --query_text")
    reset_output_dir(out_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load(clip_name, device=device)
    clip_model.eval()
    tracks = load_track_index(index_path)
    
    cam_id_to_name = {i: f"cam{i+1}" for i in range(len(videos))}
    cam_video_map = {i: videos[i] for i in range(len(videos))}
    q_hist = None
    q_aspect = None
    is_image_query = query_text.lower().endswith((".png", ".jpg", ".jpeg"))
    plate_query = None
    if is_image_query:
        yolo = YOLO("yolov8n.pt")
        q_clip, q_hist, q_aspect = embed_query_with_yolo_largest(query_text, yolo, clip_model, preprocess, device)
        constraints = {"raw": query_text, "free_text": None, "colors": [], "cameras": [], "dom_cls": None, "stop_min_sec": None, "time_range": None, "after_sec": None, "before_sec": None, "near_blast": None}
    else:
        constraints = parse_text_query(query_text, blast_time=blast_time)
        clean_q = normalize_plate(query_text)

        if is_plate_query(query_text.strip()):
            plate_query = normalize_plate(query_text)
            plate_query = clean_q
            print("PLATE SEARCH MODE:", plate_query)
        text_for_clip = constraints.get("free_text", query_text).strip()
        prompt = f"a surveillance camera image of {constraints.get('free_text', query_text)}"
        q_clip = clip_embed_text(prompt, clip_model, device)
        q_hist = None
        q_aspect = None
    print("PARSED:", constraints)
    soft_constraints = constraints.copy()
    if not is_image_query and constraints.get("free_text"):
        soft_constraints["colors"] = []
    filtered_tracks = filter_tracks(tracks, constraints, cam_id_to_name)
    results = {
        "query": {
            "text": query_text,
            "type": "image" if is_image_query else "text"
        },
        "constraints": constraints,
        "matches_by_cam": {}
    }
    candidate_tracks_for_grouping = []
    cams = list(range(len(videos)))
    for cam in cams:
        cam_name = f"cam{cam+1}"
        cand = [t for t in filtered_tracks if int(t["cam"]) == int(cam)]
        if not cand:
            results["matches_by_cam"][cam_name] = [{"status": "NOT_FOUND", "best_score": 0.0}]
            continue
        scored = []
        for t in cand:

            plate = normalize_plate(t.get("plate"))
            conf = float(t.get("plate_conf", 0.0) or 0.0)

            plate_score = 0.0

            if plate_query:
                if not plate:
                    continue

                sim = SequenceMatcher(None, plate_query, plate).ratio()

                if sim > 0.95:
                    plate_score = 1.0
                elif sim > 0.80:
                    plate_score = 0.85
                elif sim > 0.65:
                    plate_score = 0.7
                else:
                    continue
            cs = cosine_sim(q_clip, t["clip"])
            hs = 0.0
            asp = 0.0
            if q_hist is not None:
                hs = cosine_sim(q_hist, t["hist"])
            if q_aspect is None:
                asp = 0.5
            else:
                asp = aspect_sim(q_aspect, t["rep_aspect"], tol=1.2)
            color_bonus = 0.0

            if constraints.get("colors"):
                query_colors = [c.lower() for c in constraints["colors"]]
                track_color = (t.get("color_label") or "").lower()

                matched = False
                for q_color in query_colors:
                    if q_color in SIMILAR_COLORS:
                        if track_color in SIMILAR_COLORS[q_color]:
                            matched = True
                            break
                    elif track_color == q_color:
                        matched = True
                        break

                color_bonus = 0.5 if matched else 0.0
            else:
                color_bonus = 0
            if not plate_query:
                if plate and conf > 0.7:
                    plate_score = 0.1  
                else:
                    plate_score = 0.0   

            score = (
                0.60 * cs +
                0.15 * hs +
                0.10 * asp +
                0.10 * color_bonus +
                plate_score
            )
            scored.append((float(score), t))
        scored.sort(key=lambda x: x[0], reverse=True)
        seen_tracks = set()
        unique_scored = []
        for score, t in scored:
            key = (t["cam"], t["track_id"])
            if key in seen_tracks:
                continue
            unique_scored.append((score, t))
            seen_tracks.add(key)
        scored = unique_scored
        best = scored[0][0] if scored else 0.0
        if (not plate_query and best < final_thresh) or (plate_query and best == 0):
            results["matches_by_cam"][cam_name] = [{"status": "NOT_FOUND", "best_score": best}]
            continue
        items = []
        for rank, (score, t) in enumerate(scored[:min(3, topk_per_cam)], start=1):
            img_name = f"{cam_name}_rank{rank}.png"
            out_img = os.path.join(out_dir, img_name)
            plate_txt = t.get("plate")
            plate_line = f" plate={plate_txt}" if plate_txt else ""
            label = f"{cam_name} | {t.get('plate') or 'no-plate'} | {score:.2f}"
            draw_track_frame(cam_video_map[cam], t["rep_frame"], t["rep_box"], out_img, label)
            display_color = t.get("color_label")

            if constraints.get("colors"):
                q_color = constraints["colors"][0]

                if q_color == "white" and display_color in ["gray", "silver"]:
                    display_color = "white"
            item = {
                "rank": rank,
                "score": score,
                "t_start": t["t_start"],
                "t_end": t["t_end"],
                "timestamp": t.get("timestamp"),
                "location": t.get("location"),
                "image": img_name,
                "track_id": int(t["track_id"]),
                "color_label": display_color,
                "stopped_time": t.get("stopped_time"),
                "dom_cls_name": t.get("dom_cls_name"),
                "cam_name": t.get("cam_name"),
                "cam": t.get("cam"),
                "plate": t.get("plate"),
                "clip": t.get("clip").tolist() if isinstance(t.get("clip"), np.ndarray) else t.get("clip"),
                "hist": t.get("hist").tolist() if isinstance(t.get("hist"), np.ndarray) else t.get("hist"),
                "rep_aspect": t.get("rep_aspect"),
            }
            items.append(item)
            if score >= final_thresh:
                if constraints.get("dom_cls") is not None:
                    if t.get("dom_cls") != constraints["dom_cls"]:
                        continue
                cand_for_group = {
                    "cam": t.get("cam"),
                    "cam_name": t.get("cam_name"),
                    "track_id": t.get("track_id"),
                    "timestamp": t.get("timestamp"),
                    "t_start": t.get("t_start"),
                    "t_end": t.get("t_end"),
                    "image": img_name,
                    "dom_cls_name": t.get("dom_cls_name"),
                    "location": t.get("location"),          
                    "stopped_time": t.get("stopped_time"),
                    "color_label": display_color,
                    "plate": t.get("plate"),
                    "plate_conf": t.get("plate_conf", 0.0),
                    "clip": t.get("clip").tolist() if isinstance(t.get("clip"), np.ndarray) else t.get("clip"),
                    "hist": t.get("hist").tolist() if isinstance(t.get("hist"), np.ndarray) else t.get("hist"),
                    "rep_aspect": t.get("rep_aspect"),
                }
                candidate_tracks_for_grouping.append(cand_for_group)
        results["matches_by_cam"][cam_name] = items

    if group_weights is None:
        group_weights = {'clip':0.35, 'hist':0.25, 'aspect':0.05, 'color':0.30, 'plate':0.05}
    unique_tracks = {}
    for t in candidate_tracks_for_grouping:
        key = (t["cam"], t["track_id"])
        if key not in unique_tracks:
            unique_tracks[key] = t
    candidate_tracks_for_grouping = list(unique_tracks.values())
    if constraints.get("cameras") and len(constraints["cameras"]) == 1:
        groups = [[t] for t in candidate_tracks_for_grouping]
    else:
        groups = cross_camera_group_tracks(candidate_tracks_for_grouping, group_thresh=group_thresh, weights=group_weights)
    groups.sort(key=lambda g: min(_mmss_to_sec(x["t_start"]) for x in g))
    results['groups'] = groups
    grouped = group_results_by_rank(results["matches_by_cam"])
    results["grouped"] = grouped
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    make_html(results, out_dir)
    print("DONE:", os.path.join(out_dir, "results.html"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", default=None)
    ap.add_argument("--query_text", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="results_out")
    ap.add_argument("--index_dir", type=str, default="track_index")
    ap.add_argument("--blast_time", type=str, default=None)
    ap.add_argument("--rebuild_index", action="store_true")
    ap.add_argument("--final_thresh", type=float, default=0.15)
    ap.add_argument("--group_thresh", type=float, default=0.65)
    ap.add_argument("--plate_model", type=str, default=None)
    args = ap.parse_args()

    if not args.videos:
        exts = (".avi", ".mp4", ".mov", ".mkv")
        vids = [f for f in os.listdir(".") if f.lower().startswith("cam") and f.lower().endswith(exts)]
        vids.sort(key=lambda x: int(re.search(r"\d+", x).group()))
        videos = vids
    else:
        videos = args.videos

    index_path = os.path.join(args.index_dir, "tracks.json")
    ensure_dir(args.index_dir)
    if args.rebuild_index or not os.path.exists(index_path):
        build_track_index(videos, out_dir=args.index_dir, plate_model_path=args.plate_model)
    run_search(videos, index_path, query_text=args.query_text, blast_time=args.blast_time,
               out_dir=args.out_dir, final_thresh=args.final_thresh, group_thresh=args.group_thresh,
               plate_model_path=args.plate_model)
