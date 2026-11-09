"""Timeline plot: clip level on top, one row per detected label below."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .events import SceneAnalysis

FG_COLOR = "#d9622b"   # foreground: warm
BG_COLOR = "#3b7dbf"   # background: cool


def plot_timeline(scene: SceneAnalysis, title: str | None = None):
    labels = list(dict.fromkeys(e.label for e in scene.events))  # keep first-seen order
    n = max(1, len(labels))
    fig, (ax_lvl, ax_ev) = plt.subplots(
        2, 1, figsize=(10, 1.6 + 0.38 * n), sharex=True,
        gridspec_kw={"height_ratios": [1.2, max(1.0, 0.38 * n)]})

    if scene.level_db is not None:
        t = np.arange(len(scene.level_db)) / scene.fps
        ax_lvl.fill_between(t, scene.level_db.min(), scene.level_db, color="#888888", alpha=0.5, lw=0)
    ax_lvl.set_ylabel("level (dB)")
    ax_lvl.spines[["top", "right"]].set_visible(False)
    if title:
        ax_lvl.set_title(title, loc="left", fontsize=11)

    for e in scene.events:
        y = labels.index(e.label)
        color = FG_COLOR if e.role == "foreground" else BG_COLOR
        ax_ev.barh(y, e.duration, left=e.onset, height=0.7, color=color,
                   alpha=0.35 + 0.65 * min(1.0, e.confidence))
    ax_ev.set_yticks(range(len(labels)), labels if labels else ["(no events)"])
    ax_ev.invert_yaxis()
    ax_ev.set_xlim(0, scene.duration)
    ax_ev.set_xlabel("time (s)")
    ax_ev.spines[["top", "right"]].set_visible(False)
    ax_lvl.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=FG_COLOR, label="foreground"),
                           plt.Rectangle((0, 0), 1, 1, color=BG_COLOR, label="background")],
                  loc="lower right", fontsize=8, frameon=False, ncol=2,
                  bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    return fig
