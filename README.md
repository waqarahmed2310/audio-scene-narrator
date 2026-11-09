# Audio Scene Narrator

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

**Explaining complex acoustic scenes with sound event detection, foreground/background analysis and LLMs, and testing whether end-to-end audio-LLMs really understand scene structure.**

**Live demo:** https://your-app-name.streamlit.app (pick a clip under *Example*, or upload your own)

Upload a recording and the system tells you what is happening in it:

> *"Steady rain is audible throughout the recording. At 3.0 s a dog barks in the foreground, followed by a knock on a door at 7.0 s. The scene is most likely a house or garden during rainy weather."*

You can then ask questions such as *"What is in the background?"*, *"Is there speech?"* or *"What happens first?"*. Every answer is grounded in a time-stamped detection, so it can be checked.

<!-- Add a screenshot of the Streamlit app here: ![demo](docs/demo.png) -->

---

## Motivation

Large language models are increasingly connected to audio, either as **cascades** (a specialised detector produces structured output that an LLM reasons over) or as **end-to-end audio-LLMs** that take the waveform directly (e.g. Qwen2-Audio, SALMONN, Audio Flamingo). End-to-end models are convenient, but it is unclear how well they capture the **structure** of a complex scene:

* What is foreground and what is background?
* When does each sound occur, and in what order?
* Do they *hallucinate* sounds that are not there?

This project builds on earlier work on sound event tagging and foreground/background detection in complex acoustic scenes. It uses that structure as an interpretable interface between audio analysis and language models, and puts it to a small, controlled test against an end-to-end audio-LLM.

## How it works

```
audio ──► PANNs CNN14 (AudioSet, 527 classes)        frame-level probabilities, 100 fps
            │
            ▼
       event segmentation                            hysteresis thresholds, median smoothing
            │
            ▼
       foreground / background scoring               level contrast (dB) + clip coverage
            │
            ▼
       event timeline (text)  ─────►  LLM  ─────►  description / answers
```

The LLM is either a local model (Qwen2.5-3B-Instruct on a GPU) or any OpenAI-compatible API. The live demo uses `openai/gpt-oss-20b` on Groq.

**Foreground/background rule.** For each detected event:

| cue | intuition | maps to |
|---|---|---|
| level contrast | the scene gets louder while a foreground sound is active | -1 dB → 0, +6 dB → 1 |
| label coverage | ambient sounds cover most of the clip | ≤ 20 % → 1, ≥ 70 % → 0 |

`fg_score = 0.6 · contrast + 0.4 · (1 − coverage)`, foreground if ≥ 0.5. The rule is deliberately simple and explainable, and it is a drop-in slot for a learned model (`foreground_score` in `scene_narrator/events.py`).

**Grounded prompting.** The LLM never hears the audio. Its system prompt only allows statements supported by the detector output, tells it to merge redundant AudioSet labels (*Dog* / *Bark*) and to flag low-confidence detections.

## Example clips

[`examples/`](examples/) contains 8 ten-second scenes mixed from ESC-50 recordings. Each has a steady background and 2-3 foreground sounds, and in 7 of them two foreground sounds **overlap** (e.g. a car horn and a siren in the rain). The exact ground truth (event times, roles, overlaps) is in `examples/scenes.json`. `08_hard_quiet_siren.wav` is a deliberately hard case: a faint siren under heavy rain. The clips can be regenerated with `scripts/make_examples.py`.

## Benchmark

Real recordings rarely come with foreground/background labels, so the benchmark uses **synthetic scenes with exact ground truth**, mixed from [ESC-50](https://github.com/karolpiczak/ESC-50):

* 10 s scenes: one steady **background** (rain, wind, sea waves, engine, …) at -34 to -28 dBFS, plus 1-3 **foreground** events (dog, door knock, siren, cough, …) at +6 to +14 dB SNR, without overlap.
* Five templated question types with known answers:

| type | example | tests |
|---|---|---|
| `presence_yes` | Is there a dog barking in this recording? | detection |
| `presence_no` | Is there a siren in this recording? (absent) | **hallucination** |
| `background` | What is the continuous background sound? | scene structure |
| `role` | Is the rain in the foreground or the background? | **foreground/background** |
| `order` | Does the cough start before or after the knock? | temporal reasoning |

Three systems answer the same questions:

1. **Cascade:** PANNs + foreground/background scoring + Qwen2.5-3B (this project)
2. **End-to-end:** Qwen2-Audio-7B-Instruct (4-bit), which listens to the audio directly
3. **Control:** Qwen2.5-3B with no audio information, which shows the language prior alone

### Results

*30 scenes, ~140 questions. Fill in after running the notebook.*

| system | presence_yes | presence_no | background | role | order | overall |
|---|---|---|---|---|---|---|
| Cascade (PANNs + Qwen2.5-3B) | | | | | | |
| End-to-end (Qwen2-Audio-7B) | | | | | | |
| Control (LLM without audio) | | | | | | |

<!-- ![QA accuracy](results/qa_accuracy.png) -->

**Detection (cascade):** recall per role and foreground/background accuracy on detected events are saved in `results/detection_summary.csv`.

**Observations:** *(2-3 sentences: where the cascade wins, where the audio-LLM wins, typical failure cases from the error analysis.)*

## Run it

**Online:** open the [live demo](https://your-app-name.streamlit.app). No installation is needed.

**Benchmark on Colab:** open `notebooks/Audio_Scene_Narrator_Colab.ipynb`, select a T4 GPU and run the cells. The notebook builds the scenes, runs both systems and saves the results to `results/`.

**Locally:**

```bash
pip install torch            # see pytorch.org for your platform
pip install -r requirements.txt
streamlit run streamlit_app.py
```

In the sidebar, choose **API** (e.g. a free Groq key) or **Local model** (needs a GPU).

## Project structure

```
scene_narrator/
  tagger.py      PANNs wrapper: audio -> frame-level AudioSet probabilities
  events.py      segmentation, foreground/background scoring, timeline text
  narrator.py    prompts and LLM backends (local HF model or any OpenAI-compatible API)
  pipeline.py    audio -> timeline -> LLM
  plotting.py    timeline figure
  synth.py       ESC-50 scene mixer with ground truth, question templates
  evaluate.py    detection matching, answer scoring, summaries
  audio_llm.py   Qwen2-Audio baseline
streamlit_app.py   Streamlit demo
examples/          overlapping example scenes with ground truth
scripts/           script that builds the example scenes
notebooks/         Colab notebook for the full pipeline and benchmark
```

## Limitations

* The benchmark scenes are cleaner than real ones: foreground events do not overlap, and there is no reverberation mismatch. The example clips include overlaps but are not yet part of the benchmark.
* The foreground/background rule uses level and duration only. A quiet but semantically important sound (a distant siren) can be labelled background.
* PANNs labels come from AudioSet. Mapping them to the ESC-50 classes is a hand-made table.
* Templated questions measure specific skills, not open-ended scene understanding.

## Credits

* PANNs: Kong et al., *PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition*, 2020 ([code](https://github.com/qiuqiangkong/audioset_tagging_cnn), MIT)
* ESC-50: Piczak, *ESC: Dataset for Environmental Sound Classification*, 2015 (CC BY-NC 3.0). The example clips are mixed from ESC-50 recordings; see [`examples/README.md`](examples/README.md).
* Qwen2.5 and Qwen2-Audio: Qwen team, Alibaba Cloud (see the model cards for licenses)