"""Build 10-second example scenes with OVERLAPPING foreground events from ESC-50.

Each scene has one steady background sound and 2-3 foreground events. Some of the
foreground events deliberately overlap each other. The exact ground truth is written to
examples/scenes.json, and the source clips are credited in examples/README.md.

Usage:
    git clone --depth 1 https://github.com/karolpiczak/ESC-50.git
    python scripts/make_examples.py ESC-50 examples
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scene_narrator.synth import SR, _load, _loop_to, _set_level, _steady_files  # noqa: E402

DURATION = 10.0

# name, description, background class, background level (dBFS),
# foreground events as (ESC-50 class, onset in s, SNR over background in dB[, specific ESC-50 file])
SCENES = [
    ("01_rainy_street", "A car horn honks while a siren passes in the rain.",
     "rain", -30, [("car_horn", 2.0, 8, "1-24074-A-43.wav"), ("siren", 3.0, 8)]),
    ("02_farm_morning", "A rooster crows over a mooing cow, then a dog barks, all in the wind.",
     "wind", -32, [("rooster", 1.0, 10), ("cow", 2.2, 8), ("dog", 6.5, 10)]),
    ("03_busy_home", "A baby cries while someone knocks on the door, over a vacuum cleaner.",
     "vacuum_cleaner", -30, [("crying_baby", 1.5, 8), ("door_wood_knock", 3.0, 10)]),
    ("04_park_afternoon", "A dog barks and people laugh at the same time, with birds chirping.",
     "chirping_birds", -32, [("laughing", 2.0, 9), ("dog", 2.8, 10), ("clapping", 6.5, 8)]),
    ("05_night_outside", "A crow caws and a cat meows over each other, with crickets.",
     "crickets", -32, [("crow", 2.5, 10), ("cat", 3.2, 9)]),
    ("06_fireside", "Someone coughs, an alarm clock rings and they sneeze, by a crackling fire.",
     "crackling_fire", -30, [("coughing", 1.5, 10), ("clock_alarm", 2.5, 6), ("sneezing", 7.0, 10)]),
    ("07_train_ride", "A baby cries and people laugh on a moving train.",
     "train", -28, [("crying_baby", 2.0, 8), ("laughing", 3.0, 8)]),
    ("08_hard_quiet_siren", "HARD: a distant siren barely above heavy rain, with a loud knock.",
     "rain", -26, [("siren", 1.0, 2), ("door_wood_knock", 5.0, 12)]),
]

MAX_EVENT_S = 3.5


def audible_extent(fg: np.ndarray, bg: np.ndarray, hop: int = 512, margin_db: float = 0.0):
    """Sample range from the first to the last frame where `fg` is louder than `bg` (+ margin)."""
    db = lambda s: 20 * np.log10(librosa.feature.rms(y=s, frame_length=2048, hop_length=hop)[0] + 1e-9)  # noqa: E731
    fg_db, bg_db = db(fg), db(bg[: len(fg)])
    k = min(len(fg_db), len(bg_db))
    idx = np.where(fg_db[:k] > bg_db[:k] + margin_db)[0]
    if len(idx) == 0:
        return 0, len(fg)
    return int(idx[0] * hop), int(min(len(fg), (idx[-1] + 1) * hop + 1024))


def pick_clip(files: list[Path], rng: np.random.Generator, min_len: float = 1.5):
    """A random clip whose loud part (within 15 dB of its peak) lasts at least `min_len` seconds."""
    order = rng.permutation(len(files))
    best = None
    for i in order:
        x, _ = librosa.effects.trim(_load(files[i]), top_db=25)
        db = 20 * np.log10(librosa.feature.rms(y=x)[0] + 1e-9)
        loud = np.mean(db > db.max() - 15) * len(x) / SR
        if loud >= min_len:
            return files[i], x
        if best is None or loud > best[0]:
            best = (loud, files[i], x)
    return best[1], best[2]


def overlaps(events: list[dict]) -> list[dict]:
    fg = [e for e in events if e["role"] == "foreground"]
    out = []
    for i, a in enumerate(fg):
        for b in fg[i + 1:]:
            ov = min(a["offset"], b["offset"]) - max(a["onset"], b["onset"])
            if ov > 0:
                out.append({"events": [a["class"], b["class"]], "overlap_s": round(ov, 2)})
    return out


def main(esc50_root: str, out_dir: str, seed: int = 7):
    root, out = Path(esc50_root), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(root / "meta" / "esc50.csv")
    files = lambda c: [root / "audio" / f for f in meta.loc[meta.category == c, "filename"]]  # noqa: E731
    rng = np.random.default_rng(seed)
    n = int(DURATION * SR)
    scenes, used = [], []

    for name, desc, bg_cls, bg_db, fg_events in SCENES:
        bg_file = rng.choice(_steady_files(files(bg_cls), max_quiet_frac=0.05))
        bg = librosa.effects.trim(_load(bg_file), top_db=12)[0]  # drop fade-in/out so the loop has no gaps
        mix = _set_level(_loop_to(bg, n), bg_db)
        events = [{"class": bg_cls, "role": "background", "onset": 0.0, "offset": DURATION,
                   "level_dbfs": bg_db, "source": bg_file.name}]
        used.append(bg_file.name)

        for cls, onset, snr, *fixed in fg_events:
            if fixed:  # hand-picked recording
                f = root / "audio" / fixed[0]
                x = librosa.effects.trim(_load(f), top_db=25)[0]
            else:
                f, x = pick_clip(files(cls), rng)
            x = x[: int(min(MAX_EVENT_S, DURATION - onset - 0.2) * SR)]
            start = int(onset * SR)
            x = _set_level(x, bg_db + snr)
            # Keep only the part that is audible over the background, so the ground truth is honest.
            first, last = audible_extent(x, mix[start:start + len(x)])
            x = x[first:last]
            start += first
            fade = min(len(x) // 4, int(0.02 * SR))  # short fades avoid clicks
            if fade:
                x[:fade] *= np.linspace(0, 1, fade)
                x[-fade:] *= np.linspace(1, 0, fade)
            mix[start:start + len(x)] += x
            events.append({"class": cls, "role": "foreground", "onset": round(start / SR, 2),
                           "offset": round((start + len(x)) / SR, 2), "snr_db": snr, "source": f.name})
            used.append(f.name)

        peak = np.abs(mix).max()
        if peak > 0.89:
            mix *= 0.89 / peak
        path = out / f"{name}.wav"
        sf.write(path, mix, SR, subtype="PCM_16")
        events.sort(key=lambda e: e["onset"])
        scenes.append({"id": name, "file": path.name, "description": desc, "duration": DURATION,
                       "events": events, "overlaps": overlaps(events)})
        print(f"{name}: " + ", ".join(f"{e['class']} {e['onset']}-{e['offset']}s" for e in events[1:]
                                      if e["role"] == "foreground"))

    (out / "scenes.json").write_text(json.dumps(scenes, indent=2))

    # Credits: ESC-50 file names are {fold}-{freesound id}-{take}-{target}.wav
    lines = ["# Example clips", "",
             "Ten-second scenes with a steady background and 2-3 foreground sounds, several of which",
             "overlap. Pick them in the app under **Example**. Regenerate them with",
             "`python scripts/make_examples.py <path-to-ESC-50> examples`.", "",
             "| clip | scene | foreground events (s) | overlap |", "|---|---|---|---|"]
    for sc in scenes:
        fg = ", ".join(f"{e['class']} {e['onset']:.1f}-{e['offset']:.1f}" for e in sc["events"]
                       if e["role"] == "foreground")
        ov = ", ".join(f"{' + '.join(o['events'])} ({o['overlap_s']:.1f} s)" for o in sc["overlaps"]) or "none"
        lines.append(f"| `{sc['file']}` | {sc['description']} | {fg} | {ov} |")
    lines += ["", "## Credits", "",
             "The clips in this folder are mixed from recordings in the ESC-50 dataset",
             "(K. J. Piczak, *ESC: Dataset for Environmental Sound Classification*, ACM Multimedia 2015,",
             "https://github.com/karolpiczak/ESC-50). ESC-50 is licensed under",
             "[CC BY-NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/); the recordings were",
             "originally published on [Freesound](https://freesound.org). These mixes are shared under",
             "the same license, for non-commercial use only.", "",
             "Ground truth (event times, roles, levels and overlaps) is in `scenes.json`.", "",
             "| ESC-50 file | Freesound sound |", "|---|---|"]
    for fname in sorted(set(used)):
        fs_id = fname.split("-")[1]
        lines.append(f"| {fname} | https://freesound.org/s/{fs_id}/ |")
    (out / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:3])
