import os, argparse, re
from mtmc.utils import ensure_dir
from mtmc.indexing import build_track_index
from mtmc.searching import run_search

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
        if not vids:
             # Fallback if no files start with 'cam', just use the clips provided in the root
             vids = [f for f in os.listdir(".") if f.lower().startswith("clip") and f.lower().endswith(exts)]
        
        def try_int(s):
            try: return int(re.search(r"\d+", s).group())
            except: return 0
            
        vids.sort(key=try_int)
        videos = vids
    else:
        videos = args.videos

    print("Videos to process:", videos)

    index_path = os.path.join(args.index_dir, "tracks.json")
    ensure_dir(args.index_dir)
    
    if args.rebuild_index or not os.path.exists(index_path):
        build_track_index(videos, out_dir=args.index_dir, plate_model_path=args.plate_model)
    
    run_search(videos, index_path, query_text=args.query_text, blast_time=args.blast_time,
               out_dir=args.out_dir, final_thresh=args.final_thresh, group_thresh=args.group_thresh,
               plate_model_path=args.plate_model)
