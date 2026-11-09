"""Synthetic acoustic scenes with ground truth, built from ESC-50.

Each scene = one stationary background sound (looped, at a low level) + 1-3 foreground
events (trimmed to their active part, mixed at a higher level, without overlap).
Because we place every sound ourselves, we know exactly what is foreground, what is
background and when everything happens. That makes it possible to score both the
detector and the question answering of different systems.

ESC-50: https://github.com/karolpiczak/ESC-50 (CC BY-NC 3.0)
"""

from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

SR = 32000

# ESC-50 class -> (spoken name used in questions, AudioSet labels that count as a correct detection)
BACKGROUND_CLASSES = {
    "rain": ("rain", ["Rain", "Rain on surface", "Raindrop"]),
    "sea_waves": ("sea waves", ["Waves, surf", "Ocean"]),
    "crackling_fire": ("a crackling fire", ["Fire", "Crackle"]),
    "crickets": ("crickets", ["Cricket", "Insect"]),
    "wind": ("wind", ["Wind", "Wind noise (microphone)", "Rustling leaves"]),
    "chirping_birds": ("birds chirping", ["Bird", "Bird vocalization, bird call, bird song", "Chirp, tweet"]),
    "engine": ("an engine", ["Engine", "Idling", "Medium engine (mid frequency)"]),
    "train": ("a train", ["Train", "Rail transport", "Railroad car, train wagon", "Train wheels squealing"]),
    "vacuum_cleaner": ("a vacuum cleaner", ["Vacuum cleaner"]),
}
FOREGROUND_CLASSES = {
    "dog": ("a dog barking", ["Dog", "Bark", "Bow-wow"]),
    "rooster": ("a rooster", ["Chicken, rooster", "Crowing, cock-a-doodle-doo"]),
    "cat": ("a cat", ["Cat", "Meow"]),
    "cow": ("a cow", ["Cattle, bovinae", "Moo"]),
    "crow": ("a crow", ["Crow", "Caw"]),
    "door_wood_knock": ("knocking on a door", ["Knock", "Door"]),
    "glass_breaking": ("glass breaking", ["Glass", "Shatter", "Breaking"]),
    "church_bells": ("church bells", ["Church bell", "Bell"]),
    "car_horn": ("a car horn", ["Vehicle horn, car horn, honking", "Toot"]),
    "siren": ("a siren", ["Siren", "Emergency vehicle"]),
    "sneezing": ("sneezing", ["Sneeze"]),
    "clapping": ("clapping", ["Clapping", "Applause"]),
    "laughing": ("laughter", ["Laughter", "Giggle", "Chuckle, chortle", "Belly laugh", "Baby laughter"]),
    "crying_baby": ("a crying baby", ["Baby cry, infant cry", "Crying, sobbing"]),
    "coughing": ("coughing", ["Cough"]),
    "clock_alarm": ("an alarm clock", ["Alarm clock", "Alarm"]),
}
# Keywords that count as naming the background correctly in a free-text answer.
BACKGROUND_KEYWORDS = {
    "rain": ["rain"], "sea_waves": ["wave", "surf", "ocean", "sea"], "crackling_fire": ["fire", "crackl"],
    "crickets": ["cricket", "insect"], "wind": ["wind"], "chirping_birds": ["bird", "chirp", "tweet"],
    "engine": ["engine", "motor", "idling"], "train": ["train", "rail"], "vacuum_cleaner": ["vacuum"],
}
ALL_CLASSES = {**BACKGROUND_CLASSES, **FOREGROUND_CLASSES}


def _rms_db(x: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)


def _set_level(x: np.ndarray, target_db: float) -> np.ndarray:
    return x * 10 ** ((target_db - _rms_db(x)) / 20)


def _load(path: Path) -> np.ndarray:
    return librosa.load(path, sr=SR, mono=True)[0]


def _steady_files(files: list[Path], max_quiet_frac: float = 0.2) -> list[Path]:
    """Keep background clips that do not contain long silent parts (ESC-50 pads some clips)."""
    keep = []
    for f in files:
        x = _load(f)
        db = 20 * np.log10(librosa.feature.rms(y=x)[0] + 1e-9)
        if np.mean(db < db.max() - 30) <= max_quiet_frac:
            keep.append(f)
    return keep or files


def _loop_to(x: np.ndarray, n: int, fade: int = 1600) -> np.ndarray:
    out = x.copy()
    ramp = np.linspace(0, 1, fade)
    while len(out) < n:  # crossfade-loop to avoid clicks
        out = np.concatenate([out[:-fade], out[-fade:] * ramp[::-1] + x[:fade] * ramp, x[fade:]])
    return out[:n]


def build_scenes(esc50_root: str | Path, out_dir: str | Path, n_scenes: int = 30,
                 duration: float = 10.0, seed: int = 0,
                 bg_level_db: tuple[float, float] = (-34, -28),
                 fg_snr_db: tuple[float, float] = (6, 14),
                 n_foreground: tuple[int, int] = (1, 3)) -> list[dict]:
    """Mix `n_scenes` scenes into `out_dir` and write `scenes.json` with the ground truth."""
    esc50_root, out_dir = Path(esc50_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(esc50_root / "meta" / "esc50.csv")
    by_class = {c: [esc50_root / "audio" / f for f in meta.loc[meta.category == c, "filename"]]
                for c in ALL_CLASSES}
    rng = np.random.default_rng(seed)
    n = int(duration * SR)
    steady_cache: dict[str, list[Path]] = {}
    scenes = []

    for i in range(n_scenes):
        bg_cls = rng.choice(sorted(BACKGROUND_CLASSES))
        if bg_cls not in steady_cache:
            steady_cache[bg_cls] = _steady_files(by_class[bg_cls])
        bg_file = rng.choice(steady_cache[bg_cls])
        bg_db = rng.uniform(*bg_level_db)
        mix = _set_level(_loop_to(_load(bg_file), n), bg_db)
        events = [{"class": bg_cls, "role": "background", "onset": 0.0, "offset": duration,
                   "file": bg_file.name}]

        k = int(rng.integers(n_foreground[0], n_foreground[1] + 1))
        fg_classes = rng.choice(sorted(FOREGROUND_CLASSES), size=k, replace=False)
        taken: list[tuple[float, float]] = []
        for cls in fg_classes:
            f = rng.choice(by_class[cls])
            x, _ = librosa.effects.trim(_load(f), top_db=25)
            x = x[: int(3.0 * SR)]  # cap event length at 3 s
            length = len(x) / SR
            for _ in range(50):  # find a non-overlapping slot
                onset = float(rng.uniform(0.5, duration - length - 0.3))
                if all(onset + length + 0.3 < s or onset > e + 0.3 for s, e in taken):
                    break
            else:
                continue
            taken.append((onset, onset + length))
            start = int(onset * SR)
            mix[start:start + len(x)] += _set_level(x, bg_db + rng.uniform(*fg_snr_db))
            events.append({"class": str(cls), "role": "foreground", "onset": round(onset, 3),
                           "offset": round(onset + length, 3), "file": f.name})

        peak = np.abs(mix).max()
        if peak > 0.89:
            mix *= 0.89 / peak
        path = out_dir / f"scene_{i:03d}.wav"
        sf.write(path, mix, SR)
        events.sort(key=lambda e: e["onset"])
        scenes.append({"id": path.stem, "path": str(path), "duration": duration, "events": events})

    with open(out_dir / "scenes.json", "w") as fh:
        json.dump(scenes, fh, indent=2)
    return scenes


# --------------------------------------------------------------------------- questions

def _name(cls: str) -> str:
    return ALL_CLASSES[cls][0]


def make_questions(scene: dict, rng: np.random.Generator) -> list[dict]:
    """Templated questions with known answers, one per reasoning skill."""
    fg = [e for e in scene["events"] if e["role"] == "foreground"]
    bg = [e for e in scene["events"] if e["role"] == "background"][0]
    present = {e["class"] for e in scene["events"]}
    present_labels = {l for c in present for l in ALL_CLASSES[c][1]}
    qs = []

    a = fg[int(rng.integers(len(fg)))]
    qs.append({"type": "presence_yes", "answer": "yes",
               "question": f"Is there {_name(a['class'])} in this recording? Answer yes or no."})

    # Absent class that does not share detector labels with anything present (avoids ambiguous "no").
    absent = [c for c in sorted(FOREGROUND_CLASSES)
              if c not in present and not set(FOREGROUND_CLASSES[c][1]) & present_labels]
    c = absent[int(rng.integers(len(absent)))]
    qs.append({"type": "presence_no", "answer": "no",
               "question": f"Is there {_name(c)} in this recording? Answer yes or no."})

    qs.append({"type": "background", "answer": bg["class"],
               "question": "What is the continuous background sound in this recording? Answer in a few words."})

    target = bg if rng.random() < 0.5 else fg[int(rng.integers(len(fg)))]
    qs.append({"type": "role", "answer": target["role"],
               "question": f"Is {_name(target['class'])} in the foreground or in the background of this "
                           f"recording? Answer with one word: foreground or background."})

    if len(fg) >= 2:
        i, j = rng.choice(len(fg), size=2, replace=False)
        x, y = fg[i], fg[j]
        qs.append({"type": "order", "answer": "before" if x["onset"] < y["onset"] else "after",
                   "question": f"Does {_name(x['class'])} start before or after {_name(y['class'])}? "
                               f"Answer with one word: before or after."})
    for q in qs:
        q["scene"] = scene["id"]
    return qs
