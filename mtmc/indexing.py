import os, shutil, json
import cv2
import numpy as np
import torch
import clip
from ultralytics import YOLO
from tqdm import tqdm
from mtmc.utils import mmss, clamp_box, bbox_aspect, major_class
from mtmc.ocr_engine import extract_timestamp, extract_location
from mtmc.embeddings import clip_embed_bgr
from mtmc.vehicle import VEHICLE_CLASSES, get_coco_names, vehicle_color_label, hsv_hist, compute_stopped_time
from mtmc.plate import detect_plate_and_ocr, clean_plate, is_valid_plate

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
            print(f"Cannot open {vpath}")
            continue
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
            stopped_time = compute_stopped_time(tr["centers"], tr["fps"], bboxes=tr["boxes"], sample_every=sample_every)
            
            timestamp = None
            location = None
            rep_crop_color = None
            try:
                cap2 = cv2.VideoCapture(vpath)
                cap2.set(cv2.CAP_PROP_POS_FRAMES, int(rep_frame))
                ok2, fr2 = cap2.read()
                cap2.release()
                if ok2:
                    timestamp = extract_timestamp(fr2)
                    location = extract_location(fr2)
                    x1, y1, x2, y2 = rep_box
                    rep_crop_color = fr2[y1:y2, x1:x2]
            except Exception:
                pass
            
            color_label = "unknown"
            color_conf = 0.0
            if rep_crop_color is not None and rep_crop_color.size > 0:
                color_label, color_conf = vehicle_color_label(rep_crop_color)
            
            # Plate candidates sampling
            plate_candidates = []
            sample_frames = tr["frames"][::max(1, len(tr["frames"]) // 5)]
            for f in sample_frames:
                try:
                    cap_tmp = cv2.VideoCapture(vpath)
                    cap_tmp.set(cv2.CAP_PROP_POS_FRAMES, int(f))
                    ok_tmp, frame_tmp = cap_tmp.read()
                    cap_tmp.release()
                    if not ok_tmp: continue
                    idx = tr["frames"].index(f)
                    box = tr["boxes"][idx]
                    crop = frame_tmp[box[1]:box[3], box[0]:box[2]]
                    if crop is None or crop.size == 0: continue
                    h_c, w_c = crop.shape[:2]
                    crop_plate = crop[int(h_c*0.55):int(h_c*0.95), int(w_c*0.15):int(w_c*0.85)]
                    txt, p_conf = detect_plate_and_ocr(crop_plate, plate_yolo=plate_yolo)
                    if txt: plate_candidates.append((txt, p_conf))
                except Exception: continue

            plate_text = None
            plate_conf = 0.0
            if plate_candidates:
                cleaned = [(clean_plate(p), c, is_valid_plate(clean_plate(p))) for p, c in plate_candidates]
                valid_p = [x for x in cleaned if x[2]]
                if valid_p:
                    counts = {}
                    for p, c, v in valid_p: counts.setdefault(p, []).append(c)
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
