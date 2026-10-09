"""Export and prepare evidence files for roster/report/results/.

Fixes known issues from run 1:
  - matches_summary.csv: reads match["result"]["score"] correctly (was empty in run 1).
  - Copies ONLY matches referenced by probe_suite.json (avoids committing unused matches).
  - Copies fixtures.jsonl for the referenced matches.
  - behavior_distance_matrix.csv: exports declared overall_excess (kernel excess distance).
  - Copies all evidence json/csv files directly into results/.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
from pathlib import Path


def export_results(data_dir: Path, results_dir: Path, repo_root: Path):
    results_dir.mkdir(parents=True, exist_ok=True)
    matches_dst = results_dir / "matches"
    matches_dst.mkdir(parents=True, exist_ok=True)
    
    suite_path = data_dir / "probe_suite.json"
    dist_path = data_dir / "behavior_distance.json"
    strength_path = data_dir / "strength.json"
    live_path = data_dir / "live_profiles.json"
    assess_json = data_dir / "bot_assessments.json"
    assess_csv = data_dir / "all_bot_assessments.csv"
    admission_path = data_dir / "admission.json"
    
    # 1. Copy small JSON evidence files
    for src in (suite_path, dist_path, strength_path, live_path, assess_json, assess_csv, admission_path):
        if src.is_file():
            shutil.copy2(src, results_dir / src.name)
            print(f"[export] Copied {src.name} to {results_dir}")
            
    # Copy env_check files
    for env_p in (repo_root / "reports").glob("env_check_*"):
        shutil.copy2(env_p, results_dir / env_p.name)
        print(f"[export] Copied {env_p.name} to {results_dir}")
        
    if not suite_path.is_file():
        print(f"[export] Warning: {suite_path} not found.")
        return
        
    with open(suite_path, "r", encoding="utf-8") as f:
        suite = json.load(f)
        
    # 2. Copy ONLY referenced probe matches and fixtures.jsonl
    referenced_files = set()
    for p in suite["probes"]:
        fpath = Path(p["file"])
        if fpath.is_file():
            # Relative path inside raw_probes: e.g. n02/n02-s1-000003.json.gz
            rel = Path(fpath.parent.name) / fpath.name
            target = matches_dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fpath, target)
            referenced_files.add(str(rel).replace("\\", "/"))
            
    print(f"[export] Copied {len(referenced_files)} referenced match files to {matches_dst}")
    
    # Copy fixtures.jsonl matching the referenced files
    raw_fixtures = Path(suite.get("raw", "")) / "fixtures.jsonl"
    if not raw_fixtures.is_file():
        raw_fixtures = repo_root / "data" / "raw_probes_full" / "fixtures.jsonl"
        
    if raw_fixtures.is_file():
        out_fix = results_dir / "fixtures.jsonl"
        with open(raw_fixtures, "r", encoding="utf-8") as fin, open(out_fix, "w", encoding="utf-8") as fout:
            count = 0
            for line in fin:
                if not line.strip():
                    continue
                item = json.loads(line)
                rel_file = item.get("file", "").replace("\\", "/")
                if rel_file in referenced_files or not referenced_files:
                    fout.write(line)
                    count += 1
        print(f"[export] Wrote {count} fixtures to {out_fix}")
        
    # 3. Generate matches_summary.csv with CORRECT score extraction (match["result"]["score"])
    matches_csv = results_dir / "matches_summary.csv"
    with open(matches_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["match_id", "teams", "map_seed", "flag_seed", "duration_turns",
                         "seats", "scores", "winner_seat", "winner_score", "total_faults"])
        for gz_path in sorted(matches_dst.glob("n*/*.json.gz")):
            with gzip.open(gz_path, "rt", encoding="utf-8") as gf:
                m = json.load(gf)
            # Correct score extraction per engine schema
            scores = m.get("result", {}).get("score")
            if scores is None:
                scores = m.get("scores", [])
            winner_seat = int(max(range(len(scores)), key=lambda i: scores[i])) if scores else -1
            winner_score = scores[winner_seat] if winner_seat >= 0 else 0
            faults_cnt = sum(b.get("faults", 0) for b in m.get("bots", []))
            seats_str = ";".join(s.get("name", "") for s in m.get("seats", []))
            scores_str = ";".join(map(str, scores))
            writer.writerow([
                m.get("match_id"), m.get("teams"), m.get("map_seed"), m.get("flag_seed"),
                len(m.get("blocks", [])), seats_str, scores_str, winner_seat,
                winner_score, faults_cnt
            ])
    print(f"[export] Generated {matches_csv}")
    
    # 4. Generate probe_suite.csv
    probe_csv = results_dir / "probe_suite.csv"
    with open(probe_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        tags = suite.get("tags", [])
        writer.writerow(["probe_id", "teams", "map_seed", "flag_seed", "seat", "turns"] + tags)
        for p in suite["probes"]:
            row = [p["probe_id"], p["teams"], p["map_seed"], p["flag_seed"], p["seat"], p["turns"]]
            for t in tags:
                row.append(p.get("tag_counts", {}).get(t, 0))
            writer.writerow(row)
    print(f"[export] Generated {probe_csv}")
    
    # 5. Generate behavior_distance_matrix.csv (overall_excess, declared)
    if dist_path.is_file():
        with open(dist_path, "r", encoding="utf-8") as f:
            dist = json.load(f)
        matrix_csv = results_dir / "behavior_distance_matrix.csv"
        with open(matrix_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Header declares field
            writer.writerow(["bot_id (overall_excess)"] + dist["ids"])
            matrix_data = dist.get("overall_excess") or dist.get("overall")
            for b, row in zip(dist["ids"], matrix_data):
                writer.writerow([b] + [f"{x:.4f}" if x is not None else "" for x in row])
        print(f"[export] Generated {matrix_csv} (field: overall_excess)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/roster", help="Directory with current roster data")
    ap.add_argument("--results", default="roster/report/results", help="Directory to export into")
    args = ap.parse_args(argv)
    
    repo_root = Path(__file__).resolve().parent.parent
    export_results(Path(args.data), Path(args.results), repo_root)


if __name__ == "__main__":
    main()
