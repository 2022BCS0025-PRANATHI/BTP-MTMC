import re
from mtmc.utils import _mmss_to_sec
from mtmc.plate import is_plate_query, normalize_plate
from mtmc.vehicle import VEHICLE_WORDS

COLOR_WORDS = {"red", "blue", "white", "black", "silver", "gray", "grey", "yellow", "green", "orange", "purple", "pink"}
SIMILAR_COLORS = {
    "white": ["white", "gray", "silver"],
    "gray": ["gray", "silver", "white"],
    "grey": ["gray", "silver", "white"],
    "silver": ["silver", "gray", "white"],
}

LOCATION_CAMERA_MAP = {
    "subashchandrabose": "cam1",
    "kathrikkadavu": "cam2",
    "petta": "cam3",
}

STOP_WORDS = {"jn", "road", "street", "junction"}

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

    if is_plate_query(qq):
        out["is_plate_query"] = True
        out["plate_query"] = normalize_plate(qq)
        out["free_text"] = qq
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
            if t_cls != dom_cls:
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
