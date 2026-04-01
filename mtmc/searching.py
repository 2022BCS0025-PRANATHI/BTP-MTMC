import os, json, re
import cv2
import numpy as np
import torch
import clip
from ultralytics import YOLO
from difflib import SequenceMatcher
from mtmc.utils import cosine_sim, aspect_sim, reset_output_dir, _mmss_to_sec
from mtmc.plate import normalize_plate
from mtmc.embeddings import clip_embed_text, embed_query_with_yolo_largest
from mtmc.query import parse_text_query, filter_tracks, SIMILAR_COLORS
from mtmc.indexing import load_track_index
from mtmc.viz import draw_track_frame, make_html

def track_similarity(t1, t2, weights=None):
    if weights is None:
        weights = {'clip':0.55, 'hist':0.2, 'aspect':0.02, 'color':0.15, 'plate':0.08}
    
    try:
        if t1.get("timestamp") and t2.get("timestamp"):
            time1 = re.search(r"\d{2}:\d{2}:\d{2}", t1["timestamp"]).group()
            time2 = re.search(r"\d{2}:\d{2}:\d{2}", t2["timestamp"]).group()
            h1,m1,s1 = map(int,time1.split(":"))
            h2,m2,s2 = map(int,time2.split(":"))
            if abs((h1*3600+m1*60+s1)-(h2*3600+m2*60+s2)) > 600:
                return 0.0
    except:
        pass

    c1, c2 = t1.get("dom_cls"), t2.get("dom_cls")
    cls_score = 1.0 if (c1 is not None and c1 == c2) else 0.5
    
    asp1, asp2 = t1.get("rep_aspect"), t2.get("rep_aspect")
    if asp1 is not None and asp2 is not None and abs(asp1 - asp2) > 1.4:
        return 0.0

    cs = cosine_sim(np.array(t1['clip'], np.float32), np.array(t2['clip'], np.float32))
    hs = cosine_sim(np.array(t1['hist'], np.float32), np.array(t2['hist'], np.float32))
    asp = aspect_sim(t1.get('rep_aspect'), t2.get('rep_aspect'), tol=1.2)

    col1, col2 = (t1.get("color_label") or "").lower(), (t2.get("color_label") or "").lower()
    color_score = 0.5
    if col1 and col2 and col1 != "unknown" and col2 != "unknown":
        if col1 == col2: color_score = 1.0
        elif col1 in ["gray", "silver"] and col2 in ["gray", "silver"]: color_score = 0.9
        elif col1 in ["white", "silver", "gray"] and col2 in ["white", "silver", "gray"]: color_score = 0.7
        else: color_score = 0.2

    p1, p2 = normalize_plate(t1.get('plate')), normalize_plate(t2.get('plate'))
    plate_score = 0.0
    if p1 and p2 and float(t1.get('plate_conf', 0.0)) >= 0.7 and float(t2.get('plate_conf', 0.0)) >= 0.7:
        sim = SequenceMatcher(None, p1, p2).ratio()
        if sim > 0.95: plate_score = 1.0
        elif sim > 0.80: plate_score = 0.85
        elif sim > 0.60: plate_score = 0.65

    total = (weights.get('clip', 0.0) * max(0.0, cs) +
             weights.get('hist', 0.0) * max(0.0, hs) +
             weights.get('aspect', 0.0) * max(0.0, asp) +
             weights.get('color', 0.0) * color_score +
             0.1 * cls_score +
             weights.get('plate', 0.0) * plate_score)
    return float(total)

def cross_camera_group_tracks(tracks, group_thresh=0.48, weights=None):
    n = len(tracks)
    if n == 0: return []
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if int(tracks[i]['cam']) == int(tracks[j]['cam']): continue
            if track_similarity(tracks[i], tracks[j], weights=weights) >= group_thresh:
                adj[i].append(j); adj[j].append(i)
    
    visited = [False]*n
    groups = []
    for i in range(n):
        if visited[i]: continue
        stack, comp = [i], []
        visited[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True; stack.append(v)
        groups.append([tracks[idx].copy() for idx in comp])
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
            row["cams"][cam] = items[r] if (len(items) > r and items[0].get("status") != "NOT_FOUND") else None
        grouped.append(row)
    return grouped

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
    if query_text is None: raise ValueError("Provide --query_text")
    reset_output_dir(out_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load(clip_name, device=device)
    clip_model.eval()
    tracks = load_track_index(index_path)
    
    cam_id_to_name = {i: f"cam{i+1}" for i in range(len(videos))}
    cam_video_map = {i: videos[i] for i in range(len(videos))}
    
    is_image_query = query_text.lower().endswith((".png", ".jpg", ".jpeg"))
    plate_query = None
    q_hist = None
    q_aspect = None
    
    from mtmc.vehicle import hsv_hist # Avoid circular import

    if is_image_query:
        yolo = YOLO("yolov8n.pt")
        q_clip, q_hist, q_aspect = embed_query_with_yolo_largest(query_text, yolo, clip_model, preprocess, device, hsv_hist)
        constraints = {"raw": query_text, "colors": [], "cameras": [], "dom_cls": None}
    else:
        constraints = parse_text_query(query_text, blast_time=blast_time)
        if constraints.get("is_plate_query"):
            plate_query = constraints["plate_query"]
        prompt = f"a surveillance camera image of {constraints.get('free_text', query_text)}"
        q_clip = clip_embed_text(prompt, clip_model, device)

    filtered_tracks = filter_tracks(tracks, constraints, cam_id_to_name)
    results = {"query": {"text": query_text, "type": "image" if is_image_query else "text"},
               "constraints": constraints, "matches_by_cam": {}}
    
    candidate_tracks_for_grouping = []
    for cam in range(len(videos)):
        cam_name = f"cam{cam+1}"
        cand = [t for t in filtered_tracks if int(t["cam"]) == cam]
        if not cand:
            results["matches_by_cam"][cam_name] = [{"status": "NOT_FOUND", "best_score": 0.0}]
            continue
        
        scored = []
        for t in cand:
            p_score = 0.0
            if plate_query:
                p_text = normalize_plate(t.get("plate"))
                if not p_text: continue
                sim = SequenceMatcher(None, plate_query, p_text).ratio()
                if sim > 0.65: p_score = 1.0 if sim > 0.95 else (0.85 if sim > 0.80 else 0.7)
                else: continue
            
            cs = cosine_sim(q_clip, t["clip"])
            hs = cosine_sim(q_hist, t["hist"]) if q_hist is not None else 0.0
            asp = aspect_sim(q_aspect, t["rep_aspect"]) if q_aspect is not None else 0.5
            
            color_bonus = 0.0
            if constraints.get("colors"):
                t_color = (t.get("color_label") or "").lower()
                for qc in constraints["colors"]:
                    if t_color == qc or (qc in SIMILAR_COLORS and t_color in SIMILAR_COLORS[qc]):
                        color_bonus = 0.5; break
            
            if not plate_query and t.get("plate") and float(t.get("plate_conf", 0.0)) > 0.7:
                p_score = 0.1

            score = 0.60 * cs + 0.15 * hs + 0.10 * asp + 0.10 * color_bonus + p_score
            scored.append((float(score), t))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        # Deduplicate and filter
        seen_tids, unique_scored = set(), []
        for s, t in scored:
            if t["track_id"] not in seen_tids:
                unique_scored.append((s, t)); seen_tids.add(t["track_id"])
        
        if not unique_scored or (not plate_query and unique_scored[0][0] < final_thresh):
            results["matches_by_cam"][cam_name] = [{"status": "NOT_FOUND", "best_score": unique_scored[0][0] if unique_scored else 0.0}]
            continue
            
        items = []
        for rank, (score, t) in enumerate(unique_scored[:3], 1):
            img_name = f"{cam_name}_rank{rank}.png"
            label = f"{cam_name} | {t.get('plate') or 'no-plate'} | {score:.2f}"
            draw_track_frame(cam_video_map[cam], t["rep_frame"], t["rep_box"], os.path.join(out_dir, img_name), label)
            
            item = {k: t.get(k) for k in ["t_start", "t_end", "timestamp", "location", "track_id", "color_label", "stopped_time", "dom_cls_name", "cam_name", "cam", "plate", "rep_aspect"]}
            item.update({"rank": rank, "score": score, "image": img_name, "clip": t["clip"].tolist(), "hist": t["hist"].tolist()})
            items.append(item)
            if score >= final_thresh: candidate_tracks_for_grouping.append(item)
        results["matches_by_cam"][cam_name] = items

    if group_weights is None: group_weights = {'clip':0.35, 'hist':0.25, 'aspect':0.05, 'color':0.30, 'plate':0.05}
    if constraints.get("cameras") and len(constraints["cameras"]) == 1:
        groups = [[t] for t in candidate_tracks_for_grouping]
    else:
        groups = cross_camera_group_tracks(candidate_tracks_for_grouping, group_thresh=group_thresh, weights=group_weights)
    
    groups.sort(key=lambda g: min(_mmss_to_sec(x["t_start"]) for x in g))
    results['groups'] = groups
    results["grouped"] = group_results_by_rank(results["matches_by_cam"])
    
    with open(os.path.join(out_dir, "results.json"), "w") as f: json.dump(results, f, indent=2)
    make_html(results, out_dir)
    print("DONE:", os.path.join(out_dir, "results.html"))
