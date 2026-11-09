"""Frame-level sound event detection with PANNs (CNN14, trained on AudioSet, 527 classes).

The model is downloaded automatically the first time (~300 MB) into ~/panns_data.
Swap this module out for any detector that returns a (frames x classes) probability
matrix, e.g. the model from your earlier media project.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import librosa
import numpy as np

SAMPLE_RATE = 32000  # PANNs works at 32 kHz


@dataclass
class TaggerOutput:
    probs: np.ndarray       # (frames, classes), values in [0, 1]
    labels: list[str]       # class names, len == probs.shape[1]
    fps: float              # frames per second of `probs`
    audio: np.ndarray       # mono waveform at `sr`
    sr: int

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sr


def load_audio(source, sr: int = SAMPLE_RATE, max_seconds: float | None = 60.0) -> np.ndarray:
    """Load a file path or an (array, sample_rate) tuple as mono float32 at `sr`."""
    if isinstance(source, (str, os.PathLike)):
        audio, _ = librosa.load(source, sr=sr, mono=True)
    else:
        raw, in_sr = source
        raw = np.asarray(raw)
        audio = raw.astype(np.float32)
        if np.issubdtype(raw.dtype, np.integer):  # e.g. int16 from a microphone
            audio = audio / np.iinfo(raw.dtype).max
        if audio.ndim > 1:  # (samples, channels) or (channels, samples)
            audio = audio.mean(axis=1 if audio.shape[0] > audio.shape[1] else 0)
        if in_sr != sr:
            audio = librosa.resample(audio, orig_sr=in_sr, target_sr=sr)
    if max_seconds is not None:
        audio = audio[: int(max_seconds * sr)]
    if len(audio) < sr:  # the CNN needs at least ~1 s of input
        audio = np.pad(audio, (0, sr - len(audio)))
    return audio.astype(np.float32)


class PannsTagger:
    """Thin wrapper around `panns_inference.SoundEventDetection`."""

    def __init__(self, device: str | None = None):
        import torch
        from panns_inference import SoundEventDetection, labels

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = SoundEventDetection(checkpoint_path=None, device=device)
        self.labels = list(labels)

    def __call__(self, source, max_seconds: float | None = 60.0) -> TaggerOutput:
        audio = load_audio(source, max_seconds=max_seconds)
        framewise = self.model.inference(audio[None, :])[0]  # (frames, 527)
        fps = framewise.shape[0] / (len(audio) / SAMPLE_RATE)
        return TaggerOutput(probs=np.asarray(framewise), labels=self.labels,
                            fps=fps, audio=audio, sr=SAMPLE_RATE)
