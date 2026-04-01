import cv2
import numpy as np

VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck

VEHICLE_WORDS = {
    "car": 2,
    "truck": 7,
    "bus": 5,
    "motorcycle": 3,
    "bike": 3,
    "auto": 2,
    "autorickshaw": 2,
    "rickshaw": 21,
    "person": 0,
    "bicycle": 1,
}

def get_coco_names(yolo):
    try:
        names = yolo.model.names
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, (list, tuple)):
            return {i: str(n) for i, n in enumerate(names)}
    except Exception:
        pass
    return {i: f"class_{i}" for i in range(80)}

def vehicle_color_label(bgr):
    if bgr is None or bgr.size == 0:
        return "unknown", 0.0
    h, w = bgr.shape[:2]
    if h == 0 or w == 0:
        return "unknown", 0.0

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    H = hsv[..., 0].astype(np.int32)
    S = hsv[..., 1].astype(np.int32)
    V = hsv[..., 2].astype(np.int32)
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
