import os, re
import cv2

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
