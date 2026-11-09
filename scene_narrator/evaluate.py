"""Scoring for the synthetic benchmark: detection quality and question answering."""

from __future__ import annotations

import re

import pandas as pd

from .events import SceneAnalysis
from .synth import ALL_CLASSES, BACKGROUND_KEYWORDS


# --------------------------------------------------------------------------- detection

def match_events(scene_gt: dict, analysis: SceneAnalysis, min_overlap: float = 0.3) -> list[dict]:
    """For each ground-truth event: was it detected, and was its role (fg/bg) right?

    A detection counts if its AudioSet label maps to the GT class and it overlaps the GT
    event by at least `min_overlap` of the GT duration. The predicted role is the role of
    the overlapping detections weighted by overlap length.
    """
    rows = []
    for gt in scene_gt["events"]:
        ok_labels = {l.lower() for l in ALL_CLASSES[gt["class"]][1]}
        gt_dur = gt["offset"] - gt["onset"]
        overlap_by_role = {"foreground": 0.0, "background": 0.0}
        for ev in analysis.events:
            if ev.label.lower() not in ok_labels:
                continue
            ov = min(gt["offset"], ev.offset) - max(gt["onset"], ev.onset)
            if ov > 0:
                overlap_by_role[ev.role] += ov
        covered = max(overlap_by_role.values())
        detected = sum(overlap_by_role.values()) >= min_overlap * gt_dur
        pred_role = max(overlap_by_role, key=overlap_by_role.get) if detected else None
        rows.append({"scene": scene_gt["id"], "class": gt["class"], "gt_role": gt["role"],
                     "detected": detected, "pred_role": pred_role,
                     "role_correct": detected and pred_role == gt["role"],
                     "covered_s": round(covered, 2)})
    return rows


def detection_summary(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out = df.groupby("gt_role").agg(events=("detected", "size"), recall=("detected", "mean"))
    det = df[df.detected]
    out["role_accuracy (detected only)"] = det.groupby("gt_role")["role_correct"].mean()
    return out.round(3)


# --------------------------------------------------------------------------- QA scoring

def _first_choice(answer: str, choices: list[str]) -> str | None:
    """The choice word that appears first in the answer (whole-word match)."""
    text = answer.lower()
    hits = [(m.start(), c) for c in choices for m in [re.search(rf"\b{c}\b", text)] if m]
    return min(hits)[1] if hits else None


def score_answer(q: dict, answer: str) -> bool:
    t = q["type"]
    if t in ("presence_yes", "presence_no"):
        return _first_choice(answer, ["yes", "no"]) == q["answer"]
    if t == "role":
        return _first_choice(answer, ["foreground", "background"]) == q["answer"]
    if t == "order":
        return _first_choice(answer, ["before", "after"]) == q["answer"]
    if t == "background":
        text = answer.lower()
        return any(k in text for k in BACKGROUND_KEYWORDS[q["answer"]])
    raise ValueError(t)


def qa_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per system and question type, plus an overall column."""
    table = df.pivot_table(index="system", columns="type", values="correct", aggfunc="mean")
    table["overall"] = df.groupby("system")["correct"].mean()
    return table.round(3)
