import re
import cv2
import torch
import numpy as np
from mtmc.utils import clamp_box
from mtmc.ocr_engine import get_ocr

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

def detect_plate_and_ocr(rep_crop, plate_yolo=None):
    ocr = get_ocr()
    if rep_crop is None or rep_crop.size == 0:
        return None, 0.0

    plate_img = None
    best_conf = 0.0

    # 1. PLATE DETECTION
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
                if getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
                    best_box = None
                    for b in r.boxes:
                        conf = float(b.conf.item())
                        if conf > best_conf and conf > 0.4:
                            best_conf = conf
                            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                            bb = clamp_box(x1, y1, x2, y2, rep_crop.shape[1], rep_crop.shape[0])
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

    # 2. OCR PREPROCESSING & CANDIDATES
    try:
        p_img_resized = cv2.resize(plate_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        p_img_filtered = cv2.bilateralFilter(p_img_resized, 11, 17, 17)
        gray = cv2.cvtColor(p_img_filtered, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        gray = cv2.filter2D(gray, -1, kernel)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    except Exception:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) if plate_img.ndim==3 else plate_img
        thresh = gray

    candidates = []
    for img in [p_img_resized if 'p_img_resized' in locals() else plate_img, gray, thresh]:
        try:
            result = ocr.ocr(img)
            if result:
                for line in result:
                    if line:
                        for box in line:
                            text = box[1][0]
                            conf = float(box[1][1])
                            clean = re.sub(r'[^A-Z0-9]', '', text.upper())
                            
                            def looks_like_plate(text):
                                has_letter = any(c.isalpha() for c in text)
                                has_digit = any(c.isdigit() for c in text)
                                if not has_digit: return False
                                if len(text) < 4: return False
                                if has_letter and has_digit: return True
                                if text.isdigit() and len(text) >= 5: return True
                                return False

                            if looks_like_plate(clean):
                                candidates.append((clean, conf))
        except Exception:
            continue

    if not candidates:
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
        best_conf_val = max(counts[best_plate])
        best_text = best_plate
        best_score = min(1.0, 0.6 + best_conf_val)
    else:
        if cleaned:
            best_plate, best_conf_val, _ = max(cleaned, key=lambda x: x[1])
            best_text = best_plate
            best_score = min(1.0, 0.3 + best_conf_val)

    return best_text, float(best_score)
