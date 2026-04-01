import cv2
import torch
import clip
import numpy as np
from PIL import Image
from mtmc.utils import clamp_box, bbox_aspect

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

def embed_query_with_yolo_largest(query_img_path, yolo, clip_model, preprocess, device, hsv_hist_func, conf=0.25):
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
        q_hist = hsv_hist_func(bgr)
        return q_clip, q_hist, None
        
    _, _, bb = best
    x1, y1, x2, y2 = bb
    crop = bgr[y1:y2, x1:x2]
    q_clip = clip_embed_bgr(crop, clip_model, preprocess, device)
    q_hist = hsv_hist_func(crop)
    q_aspect = bbox_aspect(bb)
    return q_clip, q_hist, q_aspect
