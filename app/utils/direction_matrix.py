"""
Direction Matrix utility — in-memory indexed lookup.

Performance design:
─────────────────────────────────────────────────────────────────────────────
The direction matrix has ~5,000 rows. We build a dict index once at first use
and keep it in module-level memory. Subsequent lookups are pure dict hash
lookups: O(1), sub-millisecond.

To handle multiple uvicorn workers correctly after an upload, we check the
JSON file's modification time on every lookup. If the file has changed, we
reload and re-index — this mtime check costs ~1 microsecond (a single stat()
syscall) so it adds no meaningful latency.

File I/O only happens:
  1. On first request after server start
  2. On first request after an admin upload (detected via mtime)

Redis is NOT used — it would add a network round-trip (~1–5 ms) for data
that is effectively static between admin uploads.
─────────────────────────────────────────────────────────────────────────────
"""

import json
import os
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Storage path
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).parent.parent / "services" / "alignment" / "data"
MATRIX_FILE = _DATA_DIR / "direction_matrix.json"

# Direction → ScentDirections field name mapping
DIRECTION_TO_SCENT_KEY: dict[str, str] = {
    "Calming":    "Calming/Grounding",
    "Neutral":    "Neutral",
    "Energizing": "Elevating/Energizing",
}

# ---------------------------------------------------------------------------
# Module-level in-memory index (shared across all calls within one process)
# ---------------------------------------------------------------------------
_index: dict[tuple[str, str, str, str], str] = {}
_last_mtime: float = 0.0   # tracks when we last loaded the file


def _norm(s: str) -> str:
    """Normalise a string for case-insensitive matching."""
    return (s or "").strip().lower()


def _load_and_index() -> None:
    """
    Read direction_matrix.json from disk, build the lookup dict, and cache
    both in module-level variables.  Called automatically when needed.
    """
    global _index, _last_mtime

    if not MATRIX_FILE.exists():
        _index = {}
        _last_mtime = 0.0
        return

    with open(MATRIX_FILE, "r", encoding="utf-8") as f:
        rows: list[dict] = json.load(f)

    new_index: dict[tuple[str, str, str, str], str] = {}
    for row in rows:
        key = (
            _norm(row.get("Goal_Target", "")),
            _norm(row.get("Q2_Answer", "")),
            _norm(row.get("Q8_State", "")),
            _norm(row.get("Q10_Obstacle", "")),
        )
        direction = row.get("Direction_Result", "")
        if key[0] and direction:          # skip rows missing goal or direction
            new_index[key] = direction

    _index = new_index
    _last_mtime = MATRIX_FILE.stat().st_mtime

    print(
        f"[DirectionMatrix] Loaded and indexed {len(_index):,} rules "
        f"from {MATRIX_FILE.name}"
    )


def _ensure_fresh() -> None:
    """
    Cheap guard: stat() the file to see if it changed.
    Only reloads if the file is newer than what's in memory.
    Cost: ~1 µs (single OS stat syscall) — safe to call on every request.
    """
    global _last_mtime

    if not MATRIX_FILE.exists():
        if _index:
            _index.clear()
            _last_mtime = 0.0
        return

    current_mtime = MATRIX_FILE.stat().st_mtime
    if current_mtime != _last_mtime:
        _load_and_index()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_matrix(rows: list[dict]) -> None:
    """
    Persist the direction matrix to disk and immediately re-index it in
    memory.  Called by the upload endpoint.
    """
    global _index, _last_mtime

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(MATRIX_FILE, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)  # compact (no indent) for speed

    # Re-index immediately so this worker doesn't need to reload on next call
    _load_and_index()


def lookup_direction(
    goal: str,
    q2_answer: str,
    q8_state: str,
    q10_obstacle: str,
) -> Optional[str]:
    """
    O(1) lookup of Direction_Result for a given quiz-answer combination.

    Returns one of: 'Calming', 'Neutral', 'Energizing', or None if not found.
    """
    _ensure_fresh()          # ~1 µs mtime check; reloads only if file changed
    key = (_norm(goal), _norm(q2_answer), _norm(q8_state), _norm(q10_obstacle))
    return _index.get(key)


def resolve_direction_for_goals(
    goals: list[str],
    q2_answer: str,
    q8_state: str,
    q10_obstacle: str,
) -> str:
    """
    Resolve a single Direction for a user who may have selected multiple goals.

    Strategy (mirrors the Comprehensive Document rule R10 tie-break):
    1. Try each goal with the exact Q2/Q8/Q10 combination.
    2. If multiple goals return different directions, take a majority vote.
       Ties are broken: Calming > Neutral > Energizing (conservative default).
    3. Fall back to 'Neutral' if nothing matches at all.

    Returns one of: 'Calming', 'Neutral', 'Energizing'
    """
    results: list[str] = []

    for goal in goals:
        direction = lookup_direction(goal, q2_answer, q8_state, q10_obstacle)
        if direction:
            results.append(direction)

    if not results:
        return "Neutral"  # safe fallback when matrix not yet uploaded

    # Majority vote
    counts: dict[str, int] = {"Calming": 0, "Neutral": 0, "Energizing": 0}
    for r in results:
        if r in counts:
            counts[r] += 1

    # Tie-break order: Calming > Neutral > Energizing (most conservative first)
    tie_order = {"Calming": 0, "Neutral": 1, "Energizing": 2}
    return min(counts, key=lambda d: (-counts[d], tie_order[d]))


def get_stats() -> dict:
    """Return stats about the currently loaded index (for the status endpoint)."""
    _ensure_fresh()
    from collections import Counter
    direction_counts = Counter(_index.values())
    goals = sorted({key[0] for key in _index})
    return {
        "indexed_rules": len(_index),
        "direction_distribution": dict(direction_counts),
        "unique_goals": goals,
        "file_mtime": _last_mtime,
        "file_exists": MATRIX_FILE.exists(),
    }
