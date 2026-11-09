"""Gradio demo: upload or record audio, see the event timeline, read the description, ask questions.

From the notebook:   demo = build_demo(pipeline); demo.launch(share=True)
From a terminal:     python app.py
"""

from __future__ import annotations

import gradio as gr

from scene_narrator.events import timeline_to_text
from scene_narrator.plotting import plot_timeline

EXAMPLE_QUESTIONS = [
    "What is in the background?",
    "Which sounds stand out, and when?",
    "Is there any speech?",
    "What happens first?",
]


def build_demo(pipeline, examples: list[str] | None = None):
    def run_analysis(audio_path, high):
        if not audio_path:
            raise gr.Error("Please upload or record an audio clip first.")
        scene = pipeline.analyze(audio_path, high=high, low=high / 2)
        timeline = timeline_to_text(scene)
        description = pipeline.describe(scene) if pipeline.narrator else "(no LLM loaded)"
        state = {"timeline": timeline, "history": []}
        return plot_timeline(scene), timeline, description, state, ""

    def ask(question, state):
        if not state or not state.get("timeline"):
            raise gr.Error("Analyze a clip first.")
        if not question.strip():
            return _render(state["history"]), state, ""
        answer = pipeline.narrator.answer(state["timeline"], question, state["history"])
        state["history"].append((question, answer))
        return _render(state["history"]), state, ""

    def _render(history):
        return "\n\n".join(f"**Q:** {q}  \n**A:** {a}" for q, a in history)

    with gr.Blocks(title="Audio Scene Narrator") as demo:
        gr.Markdown("# Audio Scene Narrator\n"
                    "Sound event detection with foreground/background analysis, "
                    "explained and queried through an LLM.")
        state = gr.State({})
        with gr.Row():
            with gr.Column(scale=1):
                audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio clip")
                high = gr.Slider(0.1, 0.6, value=0.3, step=0.05, label="Detection threshold")
                btn = gr.Button("Analyze", variant="primary")
                if examples:
                    gr.Examples(examples=[[p] for p in examples], inputs=[audio])
            with gr.Column(scale=2):
                plot = gr.Plot(label="Event timeline")
                description = gr.Textbox(label="Scene description (LLM)", lines=5)
        with gr.Accordion("Detector output given to the LLM", open=False):
            timeline = gr.Textbox(lines=10, show_label=False)
        gr.Markdown("### Ask about the scene")
        with gr.Row():
            question = gr.Textbox(placeholder="e.g. " + EXAMPLE_QUESTIONS[0], show_label=False, scale=4)
            ask_btn = gr.Button("Ask", scale=1)
        gr.Examples(examples=[[q] for q in EXAMPLE_QUESTIONS], inputs=[question])
        answers = gr.Markdown()

        btn.click(run_analysis, [audio, high], [plot, timeline, description, state, answers])
        ask_btn.click(ask, [question, state], [answers, state, question])
        question.submit(ask, [question, state], [answers, state, question])
    return demo


if __name__ == "__main__":
    from scene_narrator.narrator import HFChatLLM, Narrator
    from scene_narrator.pipeline import SceneNarrator
    from scene_narrator.tagger import PannsTagger

    pipe = SceneNarrator(PannsTagger(), Narrator(HFChatLLM()))
    build_demo(pipe).launch(share=True)
