#!/usr/bin/env python3
"""
Resolve a raw run directory to its true wall-clock duration in seconds.

Strategy per directory:
  1. Find candidate log attempts for this basename (with archival-suffix
     stripping: "_FAILED", "_COMPLETE_FOR_PLOT", "_COMPLETE" are annotations
     added when results were archived into SAVED_DIRS_*, not part of the
     original slurm job's own Output: path).
  2. End time: prefer the log's own clean "End time:" footer. If the job was
     cancelled/timed out (no footer — common for long BO/LLM runs killed by
     the scheduler's wall-time limit), fall back to the results.csv mtime,
     which was validated (build_walltime_index.py) to land within a few
     seconds of the logged End time on jobs that DO have one.
  3. Start time: among attempts whose start is <= the resolved end, take the
     latest (these scripts do not checkpoint/resume — an earlier cancelled
     attempt for the same basename produced none of the data in the results.csv
     on disk today, so only the final attempt's start is relevant).

Returns None (and the caller should warn + exclude that run) if no candidate
attempt can be resolved.
"""

import json
import os
from datetime import datetime

_INDEX_PATH = os.path.join(os.path.dirname(__file__), "_walltime_index.json")
_SUFFIXES_TO_STRIP = ["_FAILED", "_COMPLETE_FOR_PLOT", "_COMPLETE"]

_index_cache = None


def load_index():
    global _index_cache
    if _index_cache is None:
        with open(_INDEX_PATH) as f:
            _index_cache = json.load(f)
    return _index_cache


def _candidates_for_basename(basename, index):
    if basename in index:
        return index[basename]
    name = basename
    for _ in range(len(_SUFFIXES_TO_STRIP)):
        stripped = False
        for suf in _SUFFIXES_TO_STRIP:
            if name.endswith(suf):
                name = name[: -len(suf)]
                stripped = True
                break
        if name in index:
            return index[name]
        if not stripped:
            break
    return []


def resolve_wall_seconds(run_dir, results_csv="results.csv", index=None):
    """Return (wall_seconds, detail_str) or (None, reason_str)."""
    if index is None:
        index = load_index()

    basename = os.path.basename(run_dir.rstrip("/"))
    candidates = _candidates_for_basename(basename, index)
    if not candidates:
        return None, f"no log match for basename '{basename}'"

    csv_path = os.path.join(run_dir, results_csv)
    if not os.path.exists(csv_path):
        return None, f"missing {results_csv}"
    mtime = datetime.utcfromtimestamp(os.path.getmtime(csv_path))

    # Prefer a candidate with its own clean End time close to (or before) mtime.
    clean = [c for c in candidates if c["end"]]
    end_dt = None
    for c in clean:
        end_dt = max(end_dt, datetime.fromisoformat(c["end"])) if end_dt else datetime.fromisoformat(c["end"])
    if end_dt is None or abs((end_dt - mtime).total_seconds()) > 3600 * 6:
        # No clean end within 6h of the data's last write — trust the mtime
        # (handles cancelled/timed-out jobs where "End time:" was never written).
        end_dt = mtime

    starts = [datetime.fromisoformat(c["start"]) for c in candidates]
    valid_starts = [s for s in starts if s <= end_dt]
    if not valid_starts:
        return None, f"all {len(candidates)} candidate start(s) are after data mtime"
    start_dt = max(valid_starts)

    wall_seconds = (end_dt - start_dt).total_seconds()
    if wall_seconds <= 0:
        return None, "non-positive resolved duration"

    detail = f"start={start_dt.isoformat()} end={end_dt.isoformat()} ({len(candidates)} candidate attempt(s))"
    return wall_seconds, detail
