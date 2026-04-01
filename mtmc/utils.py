import os, shutil, re
import numpy as np

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def reset_output_dir(path):
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
        except Exception:
            pass
    os.makedirs(path, exist_ok=True)

def mmss(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def _mmss_to_sec(ts: str) -> int:
    parts = ts.split(":")
    if len(parts) == 2:
        m, s = map(int, parts)
        return m * 60 + s
    elif len(parts) == 3:
        h, m, s = map(int, parts)
        return h * 3600 + m * 60 + s
    return 0

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
