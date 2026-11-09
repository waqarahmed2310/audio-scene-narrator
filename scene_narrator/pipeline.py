"""End-to-end pipeline: audio -> detector -> event timeline -> LLM."""

from __future__ import annotations

from .events import SceneAnalysis, analyze_probs, timeline_to_text


class SceneNarrator:
    def __init__(self, tagger, narrator=None):
        self.tagger = tagger        # e.g. PannsTagger()
        self.narrator = narrator    # e.g. Narrator(HFChatLLM()); optional

    def analyze(self, source, high: float = 0.3, low: float = 0.15, **kwargs) -> SceneAnalysis:
        out = self.tagger(source)
        return analyze_probs(out.probs, out.labels, out.fps, out.audio, out.sr,
                             high=high, low=low, **kwargs)

    def describe(self, scene: SceneAnalysis) -> str:
        return self.narrator.describe(timeline_to_text(scene))

    def ask(self, scene: SceneAnalysis, question: str, history=None) -> str:
        return self.narrator.answer(timeline_to_text(scene), question, history)
