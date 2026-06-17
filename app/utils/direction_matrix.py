"""
Direction Matrix utility — in-memory indexed lookup.

Logic:
──────
The direction matrix encodes every combination of quiz answers as a Rule_ID
like `DM_Q9_O1_Q2_O1_Q8_O1_Q10_O5`, where each segment names the question
and the option the user picked.

For each combination, three scores are pre-computed (Calming, Energizing,
Neutral). The highest score determines `Direction_Result`.  That field is
already the authoritative answer — we never re-compute it; we just look it up.

Lookup strategy:
1. Build a primary index keyed by (Q9_Goal_ID, Q2_Answer_ID, Q8_State_ID,
   Q10_Obstacle_ID) → Direction_Result.
2. Build reverse text→ID maps from the matrix so quiz answer text can be
   translated to the canonical option IDs (Q9_O1, Q2_O3, etc.).
3. To resolve a user's direction: translate each answer text to its ID, look
   up the combination — O(1), no fuzzy matching.

Performance:
───────────
File is loaded once, indexed in memory.  Subsequent calls cost a single
stat() syscall (~1 µs) to detect if the file changed, then a dict lookup.
"""

import json
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Storage path
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).parent.parent / "services" / "alignment" / "data"
MATRIX_FILE = _DATA_DIR / "direction_matrix.json"

# Direction → ScentDirections field name (used in affirmation schema)
DIRECTION_TO_SCENT_KEY: dict[str, str] = {
    "Calming":    "Calming/Grounding",
    "Neutral":    "Neutral",
    "Energizing": "Elevating/Energizing",
}

# ---------------------------------------------------------------------------
# Module-level indexes (populated once, refreshed on file change)
# ---------------------------------------------------------------------------

# Primary lookup: (Q9_Goal_ID, Q2_Answer_ID, Q8_State_ID, Q10_Obstacle_ID) → Direction_Result
_id_index: dict[tuple[str, str, str, str], str] = {}

# Reverse text→ID maps per question type (lower-cased text → canonical ID)
# e.g. _text_to_id["q9"]["discipline and focus"] = "Q9_O1"
_text_to_id: dict[str, dict[str, str]] = {
    "q9":  {},   # Goal_Target → Q9_Goal_ID
    "q2":  {},   # Q2_Answer   → Q2_Answer_ID
    "q8":  {},   # Q8_State    → Q8_State_ID
    "q10": {},   # Q10_Obstacle → Q10_Obstacle_ID
}

_last_mtime: float = 0.0


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _load_and_index() -> None:
    """Read the JSON file, build the ID index and text→ID maps."""
    global _id_index, _text_to_id, _last_mtime

    if not MATRIX_FILE.exists():
        _id_index = {}
        _text_to_id = {"q9": {}, "q2": {}, "q8": {}, "q10": {}}
        _last_mtime = 0.0
        return

    with open(MATRIX_FILE, "r", encoding="utf-8") as f:
        rows: list[dict] = json.load(f)

    new_id_index: dict[tuple[str, str, str, str], str] = {}
    new_text: dict[str, dict[str, str]] = {"q9": {}, "q2": {}, "q8": {}, "q10": {}}

    for row in rows:
        q9_id  = row.get("Q9_Goal_ID", "")
        q2_id  = row.get("Q2_Answer_ID", "")
        q8_id  = row.get("Q8_State_ID", "")
        q10_id = row.get("Q10_Obstacle_ID", "")
        result = row.get("Direction_Result", "")

        if not (q9_id and q2_id and q8_id and q10_id and result):
            continue  # skip incomplete rows

        # Primary index (ID-based)
        new_id_index[(q9_id, q2_id, q8_id, q10_id)] = result

        # Reverse text→ID maps (built from matrix — guaranteed to match)
        goal_text = _norm(row.get("Goal_Target", ""))
        q2_text   = _norm(row.get("Q2_Answer", ""))
        q8_text   = _norm(row.get("Q8_State", ""))
        q10_text  = _norm(row.get("Q10_Obstacle", ""))

        if goal_text:
            new_text["q9"][goal_text]  = q9_id
        if q2_text:
            new_text["q2"][q2_text]    = q2_id
        if q8_text:
            new_text["q8"][q8_text]    = q8_id
        if q10_text:
            new_text["q10"][q10_text]  = q10_id

    _id_index   = new_id_index
    _text_to_id = new_text
    _last_mtime = MATRIX_FILE.stat().st_mtime

    print(
        f"[DirectionMatrix] Loaded and indexed {len(_id_index):,} rules "
        f"from {MATRIX_FILE.name} "
        f"({len(new_text['q9'])} goals, {len(new_text['q2'])} Q2 options, "
        f"{len(new_text['q8'])} Q8 states, {len(new_text['q10'])} Q10 obstacles)"
    )


def _ensure_fresh() -> None:
    """stat() the file (~1 µs). Reload only if mtime changed."""
    if not MATRIX_FILE.exists():
        if _id_index:
            _id_index.clear()
        return
    current_mtime = MATRIX_FILE.stat().st_mtime
    if current_mtime != _last_mtime:
        _load_and_index()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_matrix(rows: list[dict]) -> None:
    """Persist the matrix to disk and re-index immediately."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(MATRIX_FILE, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)  # compact (no indent) — faster
    _load_and_index()


def lookup_direction(
    goal_text: str,
    q2_text: str,
    q8_text: str,
    q10_text: str,
) -> Optional[str]:
    """
    Resolve Direction_Result for a given combination of quiz answer texts.

    Steps:
    1. Translate each answer text to its canonical option ID using the
       reverse maps built from the matrix.
    2. Look up (Q9_Goal_ID, Q2_Answer_ID, Q8_State_ID, Q10_Obstacle_ID)
       in the primary index.
    3. Return the pre-computed Direction_Result ('Calming'/'Neutral'/'Energizing').

    Returns None if the combination is not found (matrix not uploaded, or
    answers that don't exist in the matrix).
    """
    _ensure_fresh()

    q9_id  = _text_to_id["q9"].get(_norm(goal_text))
    q2_id  = _text_to_id["q2"].get(_norm(q2_text))
    q8_id  = _text_to_id["q8"].get(_norm(q8_text))
    q10_id = _text_to_id["q10"].get(_norm(q10_text))

    if not (q9_id and q2_id and q8_id and q10_id):
        # One or more answers didn't map to a known ID
        missing = {
            k: v for k, v in
            {"goal": q9_id, "q2": q2_id, "q8": q8_id, "q10": q10_id}.items()
            if not v
        }
        print(f"[DirectionMatrix] WARNING — no ID found for: {missing} "
              f"(goal={goal_text!r}, q2={q2_text!r}, q8={q8_text!r}, q10={q10_text!r})")
        return None

    return _id_index.get((q9_id, q2_id, q8_id, q10_id))


def resolve_direction_for_goals(
    goals: list[str],
    q2_answer: str,
    q8_state: str,
    q10_obstacle: str,
) -> str:
    """
    Resolve direction for a user who may have selected multiple goals.

    For each goal, look up the Direction_Result from the pre-computed matrix.
    If multiple goals return different directions, take a majority vote.
    Tie-break: Calming > Neutral > Energizing (conservative default).
    Falls back to 'Neutral' if nothing matches (matrix not uploaded yet).
    """
    results: list[str] = []

    for goal in goals:
        direction = lookup_direction(goal, q2_answer, q8_state, q10_obstacle)
        if direction:
            results.append(direction)

    if not results:
        return "Neutral"

    counts: dict[str, int] = {"Calming": 0, "Neutral": 0, "Energizing": 0}
    for r in results:
        if r in counts:
            counts[r] += 1

    # Tie-break: Calming first (most conservative)
    tie_order = {"Calming": 0, "Neutral": 1, "Energizing": 2}
    return min(counts, key=lambda d: (-counts[d], tie_order[d]))


def get_stats() -> dict:
    """Return stats about the currently loaded index."""
    _ensure_fresh()
    from collections import Counter
    direction_counts = Counter(_id_index.values())
    goals = sorted({_norm(k) for k in _text_to_id["q9"].keys()})
    return {
        "indexed_rules": len(_id_index),
        "direction_distribution": dict(direction_counts),
        "unique_goals": goals,
        "file_mtime": _last_mtime,
        "file_exists": MATRIX_FILE.exists(),
        "text_id_map_sizes": {
            "q9_goals": len(_text_to_id["q9"]),
            "q2_answers": len(_text_to_id["q2"]),
            "q8_states": len(_text_to_id["q8"]),
            "q10_obstacles": len(_text_to_id["q10"]),
        },
    }
