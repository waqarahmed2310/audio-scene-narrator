"""Turn frame-level detector output into a structured event timeline.

Steps
1. Smooth each class's probability curve over time.
2. Segment events with hysteresis thresholding (start above `high`, extend while above `low`).
3. Score each event as FOREGROUND or BACKGROUND from two interpretable cues:
   - level contrast: how much louder the clip is while the event is active (dB)
   - label coverage: how much of the clip the sound covers (ambient sounds are long and steady)
4. Render the timeline as compact text that an LLM can reason over.

The foreground/background rule is deliberately simple and explainable. It is the natural
place to plug in a learned model (e.g. from earlier foreground/background detection work).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import librosa
import numpy as np
from scipy.ndimage import median_filter

# AudioSet labels that describe a broad category rather than a specific sound.
GENERIC_LABELS = {
    "Animal", "Domestic animals, pets", "Livestock, farm animals, working animals",
    "Wild animals", "Vehicle", "Sound effect", "Noise", "Environmental noise",
    "Mechanisms", "Liquid", "Silence", "Field recording", "Fowl",
}
# AudioSet labels that describe the acoustic environment, not a sound source.
ENVIRONMENT_LABELS = {
    "Inside, small room", "Inside, large room or hall", "Inside, public space",
    "Outside, urban or manmade", "Outside, rural or natural",
}


@dataclass
class Event:
    label: str
    onset: float            # seconds
    offset: float           # seconds
    confidence: float       # mean detector probability inside the event
    peak: float             # max detector probability inside the event
    contrast_db: float      # level inside the event minus level elsewhere in the clip
    label_coverage: float   # fraction of the clip covered by this label (all its segments)
    fg_score: float         # 0 = clearly background, 1 = clearly foreground
    role: str               # "foreground" | "background"

    @property
    def duration(self) -> float:
        return self.offset - self.onset


@dataclass
class SceneAnalysis:
    duration: float
    events: list[Event]
    environment: list[tuple[str, float]] = field(default_factory=list)
    level_db: np.ndarray | None = None   # per-frame level for plotting
    fps: float = 100.0

    def to_dict(self) -> dict:
        return {
            "duration": round(self.duration, 2),
            "environment": [{"label": l, "score": round(s, 3)} for l, s in self.environment],
            "events": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(e).items()}
                       for e in self.events],
        }


# --------------------------------------------------------------------------- helpers

def frame_level_db(audio: np.ndarray, sr: int, n_frames: int, floor_db: float = 60.0) -> np.ndarray:
    """Per-frame RMS level in dB, resampled to `n_frames` frames, floored at max - floor_db."""
    hop = max(1, len(audio) // n_frames)
    rms = librosa.feature.rms(y=audio, frame_length=max(1024, 2 * hop), hop_length=hop, center=True)[0]
    db = 20 * np.log10(rms + 1e-8)
    db = np.maximum(db, db.max() - floor_db)
    x_old = np.linspace(0, 1, len(db))
    x_new = np.linspace(0, 1, n_frames)
    return np.interp(x_new, x_old, db)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start/end indices (end exclusive) of consecutive True runs."""
    padded = np.concatenate([[False], mask, [False]]).astype(int)
    diff = np.diff(padded)
    return list(zip(np.where(diff == 1)[0], np.where(diff == -1)[0]))


def _hysteresis_segments(p: np.ndarray, high: float, low: float,
                         merge_gap: int, min_len: int) -> list[tuple[int, int]]:
    segs = [(s, e) for s, e in _runs(p >= low) if p[s:e].max() >= high]
    merged: list[list[int]] = []
    for s, e in segs:
        if merged and s - merged[-1][1] <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if e - s >= min_len]


def foreground_score(contrast_db: float, label_coverage: float) -> float:
    """Combine the two cues into a 0..1 score (>= 0.5 means foreground).

    contrast: -1 dB -> 0, +6 dB -> 1   (does the scene get louder when the sound is active?)
    coverage: <=20% -> 1, >=70% -> 0   (short sounds are more likely foreground)
    """
    contrast_term = float(np.clip((contrast_db + 1.0) / 7.0, 0.0, 1.0))
    coverage_term = 1.0 - float(np.clip((label_coverage - 0.2) / 0.5, 0.0, 1.0))
    return 0.6 * contrast_term + 0.4 * coverage_term


# --------------------------------------------------------------------------- main entry

def analyze_probs(probs: np.ndarray, labels: list[str], fps: float,
                  audio: np.ndarray, sr: int,
                  high: float = 0.3, low: float = 0.15,
                  smooth_s: float = 0.25, merge_gap_s: float = 0.3, min_dur_s: float = 0.2,
                  max_labels: int = 10, fg_threshold: float = 0.5) -> SceneAnalysis:
    """Convert a (frames x classes) probability matrix into a SceneAnalysis."""
    n_frames = probs.shape[0]
    duration = n_frames / fps
    k = max(1, int(round(smooth_s * fps)) | 1)  # odd kernel size
    smoothed = median_filter(probs, size=(k, 1), mode="nearest")
    level = frame_level_db(audio, sr, n_frames)

    # Acoustic environment: clip-level average of the environment labels.
    environment = []
    for i, lab in enumerate(labels):
        if lab in ENVIRONMENT_LABELS:
            score = float(smoothed[:, i].mean())
            if score >= low:
                environment.append((lab, score))
    environment.sort(key=lambda x: -x[1])

    # Candidate sound classes, strongest first.
    peaks = smoothed.max(axis=0)
    candidates = [i for i in np.argsort(-peaks)
                  if peaks[i] >= high and labels[i] not in GENERIC_LABELS
                  and labels[i] not in ENVIRONMENT_LABELS][:max_labels]

    events: list[Event] = []
    for c in candidates:
        p = smoothed[:, c]
        segs = _hysteresis_segments(p, high, low,
                                    merge_gap=int(merge_gap_s * fps), min_len=int(min_dur_s * fps))
        if not segs:
            continue
        active = np.zeros(n_frames, dtype=bool)
        for s, e in segs:
            active[s:e] = True
        coverage = active.mean()
        for s, e in segs:
            inside = level[s:e].mean()
            outside_mask = ~active
            # If the sound is (almost) everywhere there is nothing to compare against.
            contrast = float(inside - level[outside_mask].mean()) if outside_mask.mean() > 0.05 else 0.0
            score = foreground_score(contrast, coverage)
            events.append(Event(
                label=labels[c], onset=s / fps, offset=e / fps,
                confidence=float(probs[s:e, c].mean()), peak=float(probs[s:e, c].max()),
                contrast_db=contrast, label_coverage=float(coverage), fg_score=score,
                role="foreground" if score >= fg_threshold else "background",
            ))

    events.sort(key=lambda ev: (ev.onset, -ev.confidence))
    return SceneAnalysis(duration=duration, events=events, environment=environment,
                         level_db=level, fps=fps)


def timeline_to_text(scene: SceneAnalysis) -> str:
    """Compact, LLM-friendly description of the detector output."""
    lines = [f"Clip duration: {scene.duration:.1f} s"]
    if scene.environment:
        env = ", ".join(f"{l} ({s:.2f})" for l, s in scene.environment[:3])
        lines.append(f"Acoustic environment (clip-level guess): {env}")
    if not scene.events:
        lines.append("Detected sound events: none above threshold.")
        return "\n".join(lines)
    lines.append("Detected sound events (start-end | label | role | confidence | level vs. rest of clip):")
    for e in scene.events:
        lines.append(f"- {e.onset:5.2f}-{e.offset:5.2f} s | {e.label} | {e.role.upper()} | "
                     f"conf {e.confidence:.2f} | {e.contrast_db:+.1f} dB")
    return "\n".join(lines)
