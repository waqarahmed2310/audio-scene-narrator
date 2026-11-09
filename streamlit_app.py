"""Streamlit demo: upload, record or pick a clip, see the event timeline, read the description, ask questions.

Run:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from scene_narrator import narrator as nar
from scene_narrator.events import analyze_probs, timeline_to_text
from scene_narrator.plotting import plot_timeline

st.set_page_config(page_title="Audio Scene Narrator", page_icon="🎧", layout="wide")

# Provider -> (base URL, default model). All of them speak the OpenAI chat-completions format.
API_PRESETS = {
    "Groq": ("https://api.groq.com/openai/v1", "openai/gpt-oss-20b"),
    "Hugging Face": ("https://router.huggingface.co/v1", "Qwen/Qwen2.5-7B-Instruct"),
    "OpenAI": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "Ollama (local)": ("http://localhost:11434/v1", "qwen2.5:3b"),
}
QUICK_QUESTIONS = [
    "What is in the background?",
    "Which sounds stand out, and when?",
    "Is there any speech?",
    "What happens first?",
]
AUDIO_TYPES = ["wav", "mp3", "flac", "ogg", "m4a"]
EXAMPLES_DIR = Path(os.environ.get("EXAMPLES_DIR", "examples"))


def secret(name: str, default: str = "") -> str:
    """Read from .streamlit/secrets.toml (or Streamlit Cloud secrets), then environment variables."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # no secrets file
        pass
    return os.environ.get(name, default)


def has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


@st.cache_resource(show_spinner="Loading the sound event detector (first run downloads ~300 MB)...")
def get_tagger():
    from scene_narrator.tagger import PannsTagger
    return PannsTagger()


@st.cache_resource(show_spinner="Loading the local language model...")
def get_local_llm(model_id: str):
    return nar.HFChatLLM(model_id)


@st.cache_data(show_spinner=False, max_entries=20)
def run_detector(audio_bytes: bytes, suffix: str):
    """Run the CNN once per clip. Changing the threshold afterwards only re-runs the cheap post-processing."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        path = f.name
    try:
        out = get_tagger()(path)
    finally:
        os.unlink(path)
    return out.probs, out.labels, out.fps, out.audio, out.sr


# --------------------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("Settings")
    high = st.slider("Detection threshold", 0.10, 0.60, 0.30, 0.05,
                     help="Minimum detector confidence for a sound to count. Lower finds more sounds "
                          "but also more false detections. An event continues while the confidence "
                          "stays above half of this value.")

    st.subheader("Language model")
    gpu = has_gpu()
    backend = st.radio("Backend", ["API", "Local model"], index=1 if gpu else 0, horizontal=True,
                       help="API: any OpenAI-compatible endpoint (Groq and Hugging Face have free tiers). "
                            "Local: runs a Hugging Face model on this machine and needs a GPU.")
    llm, llm_name = None, ""
    if backend == "Local model":
        model_id = st.text_input("Model", "Qwen/Qwen2.5-3B-Instruct")
        if not gpu:
            st.warning("No GPU found. A local model will be very slow or run out of memory.")
        if st.toggle("Load model", value=gpu):
            llm, llm_name = get_local_llm(model_id), model_id
    else:
        secret_key = secret("LLM_API_KEY")
        if secret_key:  # owner-configured deployment: never send the key to the browser
            base_url = secret("LLM_BASE_URL", API_PRESETS["Groq"][0])
            model = secret("LLM_MODEL", API_PRESETS["Groq"][1])
            st.caption(f"Using the configured API: `{model}`")
            api_key = secret_key
        else:
            preset = st.selectbox("Provider", list(API_PRESETS))
            base_url = st.text_input("Base URL", API_PRESETS[preset][0], key=f"url_{preset}")
            model = st.text_input("Model", API_PRESETS[preset][1], key=f"model_{preset}")
            api_key = st.text_input("API key", type="password",
                                    help="Only kept for this session. Not needed for Ollama.")
        if api_key or "localhost" in base_url:
            llm, llm_name = nar.OpenAICompatibleLLM(model, base_url, api_key or "none"), model
        else:
            st.info("Enter an API key to enable the description and chat.")

    st.divider()
    st.caption("Detector: PANNs CNN14 (AudioSet, 527 classes). Only the first 60 s of a clip are analysed. "
               "The language model never hears the audio, it only reads the detector output.")

# --------------------------------------------------------------------------- input

st.title("🎧 Audio Scene Narrator")
st.write("Sound event detection with foreground/background analysis, explained and queried through a language model.")

examples = sorted(p for p in EXAMPLES_DIR.glob("*") if p.suffix.lstrip(".").lower() in AUDIO_TYPES) \
    if EXAMPLES_DIR.is_dir() else []
sources = ["Upload", "Record"] + (["Example"] if examples else [])
source = st.radio("Audio source", sources, horizontal=True, label_visibility="collapsed")

audio_bytes, name = None, ""
if source == "Upload":
    f = st.file_uploader("Audio file", type=AUDIO_TYPES)
    if f:
        audio_bytes, name = f.getvalue(), f.name
elif source == "Record":
    f = st.audio_input("Record a clip")
    if f:
        audio_bytes, name = f.getvalue(), "recording.wav"
else:
    choice = st.selectbox("Example clip", examples, format_func=lambda p: p.name)
    audio_bytes, name = choice.read_bytes(), choice.name

if not audio_bytes:
    st.info("Upload, record or choose a clip to start.")
    st.stop()

if source != "Record":
    st.audio(audio_bytes)

# --------------------------------------------------------------------------- analysis

clip_id = hashlib.md5(audio_bytes).hexdigest()
try:
    with st.spinner("Detecting sound events..."):
        probs, labels, fps, audio, sr = run_detector(audio_bytes, Path(name).suffix or ".wav")
except Exception as e:
    st.error(f"Could not analyse this clip: {e}")
    st.stop()

scene = analyze_probs(probs, labels, fps, audio, sr, high=high, low=high / 2)
timeline = timeline_to_text(scene)

# New clip: clear the chat. New clip, threshold or model: rewrite the description.
if st.session_state.get("clip_id") != clip_id:
    st.session_state.update(clip_id=clip_id, history=[])
if st.session_state.get("desc_key") != (clip_id, high, llm_name):
    st.session_state.update(desc_key=(clip_id, high, llm_name), description=None)

left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("Event timeline")
    fig = plot_timeline(scene)
    st.pyplot(fig)
    plt.close(fig)
    if scene.events:
        table = pd.DataFrame([{
            "sound": e.label, "start (s)": round(e.onset, 2), "end (s)": round(e.offset, 2),
            "role": e.role, "confidence": round(e.confidence, 2), "level vs. rest (dB)": round(e.contrast_db, 1),
        } for e in scene.events])
        st.dataframe(table, hide_index=True)
    else:
        st.warning("No sound events above the threshold. Try lowering it in the sidebar.")
    with st.expander("Detector output given to the language model"):
        st.code(timeline, language=None)

with right:
    st.subheader("Scene description")
    if llm is None:
        st.info("Configure a language model in the sidebar to get a description.")
    else:
        if st.session_state.description is None:
            try:
                with st.spinner("Writing the description..."):
                    st.session_state.description = nar.Narrator(llm).describe(timeline)
            except Exception as e:
                st.error(f"Language model request failed: {e}")
        if st.session_state.description:
            st.write(st.session_state.description)
            if st.button("Regenerate", icon="🔄"):
                st.session_state.description = None
                st.rerun()

# --------------------------------------------------------------------------- chat

st.divider()
st.subheader("Ask about the scene")
if llm is None:
    st.caption("Configure a language model in the sidebar to ask questions.")
    st.stop()

quick = st.pills("Suggestions", QUICK_QUESTIONS, label_visibility="collapsed", key=f"pills_{len(st.session_state.history)}")
typed = st.chat_input("Ask a question about this recording")
question = typed or quick

for q, a in st.session_state.history:
    st.chat_message("user").write(q)
    st.chat_message("assistant").write(a)

if question:
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                answer = nar.Narrator(llm).answer(timeline, question, st.session_state.history)
            st.write(answer)
            st.session_state.history.append((question, answer))
            if quick:  # clear the selected suggestion (its key depends on the history length)
                st.rerun()
        except Exception as e:
            st.error(f"Language model request failed: {e}")

if st.session_state.history and st.button("Clear chat"):
    st.session_state.history = []
    st.rerun()
