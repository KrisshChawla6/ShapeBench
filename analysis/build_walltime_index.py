#!/usr/bin/env python3
"""
Build a lookup index of per-run wall-clock time from slurm .out logs.

Scans logs/jack/*.out. For each job with a parseable "Output:" line (the run's
result directory), records every (start, end-or-None) attempt under that run's
basename — including jobs that never got a clean "End time:" footer (killed by
the scheduler's time limit, cancelled after convergence stagnation, etc.).
Multiple attempts can map to the same basename (requeues); resolving which
attempt actually produced the data on disk today is done at lookup time
(see walltime_lookup.py), using each run's own results.csv mtime as a
completion proxy when no clean end time was logged.

Output: analysis/_walltime_index.json
    { "<run_dir_basename>": [ {"log_file":..., "start": iso, "end": iso|null}, ... ] }
"""

import json
import os
import re
from datetime import datetime

LOG_DIR = "/scratch/ShapeEvolve/logs/jack"
OUT_PATH = os.path.join(os.path.dirname(__file__), "_walltime_index.json")

DATE_FMT = "%a %b %d %H:%M:%S UTC %Y"


def parse_date(line):
    cleaned = re.sub(r"\s+", " ", line.strip())
    return datetime.strptime(cleaned, DATE_FMT)


def main():
    index = {}
    n_scanned = 0
    n_with_output = 0
    n_with_start = 0

    for fname in sorted(os.listdir(LOG_DIR)):
        if not fname.endswith(".out"):
            continue
        n_scanned += 1
        path = os.path.join(LOG_DIR, fname)
        start_dt = None
        end_dt = None
        output_path = None

        with open(path, errors="replace") as f:
            for line in f:
                if line.startswith("Date:") and start_dt is None:
                    try:
                        start_dt = parse_date(line[len("Date:"):])
                    except ValueError:
                        pass
                elif line.startswith("Output:") and output_path is None:
                    output_path = line[len("Output:"):].strip()
                elif line.startswith("End time:"):
                    try:
                        end_dt = parse_date(line[len("End time:"):])
                    except ValueError:
                        pass

        if output_path is None:
            continue
        n_with_output += 1
        if start_dt is None:
            continue
        n_with_start += 1

        basename = os.path.basename(output_path.rstrip("/"))
        index.setdefault(basename, []).append({
            "log_file": fname,
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat() if end_dt else None,
        })

    with open(OUT_PATH, "w") as f:
        json.dump(index, f, indent=1, sort_keys=True)

    n_clean_end = sum(1 for entries in index.values() for e in entries if e["end"])
    print(f"Scanned {n_scanned} .out files")
    print(f"  {n_with_output} had a parseable Output: line")
    print(f"  {n_with_start} also had a parseable Date: (start)")
    print(f"  {len(index)} unique run-dir basenames indexed"
          f" ({sum(len(v) for v in index.values())} total attempts,"
          f" {n_clean_end} with a clean End time:)")
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
