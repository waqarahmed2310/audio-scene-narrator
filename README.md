# Audio Scene Narrator

**Explaining complex acoustic scenes with sound event detection, foreground/background analysis and LLMs, and testing whether end-to-end audio-LLMs really understand scene structure.**

Upload a recording and the system tells you what is happening in it:

> *"Steady rain is audible throughout the recording. At 3.0 s a dog barks in the foreground, followed by a knock on a door at 7.0 s. The scene is most likely a house or garden during rainy weather."*

You can then ask questions such as *"What is in the background?"*, *"Is there speech?"* or *"What happens first?"*. Every answer is grounded in a time-stamped detection, so it can be checked.

<!-- Add a screenshot of the Gradio demo here: ![demo](docs/demo.png) -->

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
       event timeline (text)  ─────►  LLM (Qwen2.5-3B-Instruct)  ─────►  description / answers
```

**Foreground/background rule.** For each detected event:

| cue | intuition | maps to |
|---|---|---|
| level contrast | the scene gets louder while a foreground sound is active | -1 dB → 0, +6 dB → 1 |
| label coverage | ambient sounds cover most of the clip | ≤ 20 % → 1, ≥ 70 % → 0 |

`fg_score = 0.6 · contrast + 0.4 · (1 − coverage)`, foreground if ≥ 0.5. The rule is deliberately simple and explainable, and it is a drop-in slot for a learned model (`foreground_score` in `scene_narrator/events.py`).

**Grounded prompting.** The LLM never hears the audio. Its system prompt only allows statements supported by the detector output, tells it to merge redundant AudioSet labels (*Dog* / *Bark*) and to flag low-confidence detections.

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

**Colab (recommended):** open `notebooks/Audio_Scene_Narrator_Colab.ipynb`, select a T4 GPU and run all cells (~30-40 min including downloads). The notebook builds the scenes, runs both systems, saves the results to `results/`, and launches the Gradio demo.

**Locally** (GPU with ≥ 8 GB recommended):

```bash
pip install torch            # see pytorch.org for your platform
pip install -r requirements.txt
python app.py                # demo
```

```python
from scene_narrator.tagger import PannsTagger
from scene_narrator.narrator import HFChatLLM, Narrator
from scene_narrator.pipeline import SceneNarrator

pipe = SceneNarrator(PannsTagger(), Narrator(HFChatLLM()))
scene = pipe.analyze("my_recording.wav")
print(pipe.describe(scene))
print(pipe.ask(scene, "What is in the background?"))
```

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
app.py           Gradio demo
notebooks/       Colab notebook for the full pipeline and benchmark
```

## Limitations

* Synthetic scenes are cleaner than real ones (no overlapping foreground events, no reverberation mismatch).
* The foreground/background rule uses level and duration only. A quiet but semantically important sound (a distant siren) can be labelled background.
* PANNs labels come from AudioSet. Mapping them to the ESC-50 classes is a hand-made table.
* Templated questions measure specific skills, not open-ended scene understanding.

## Toward a Master's thesis

This prototype points to a research question: **How well do audio-language models capture the structure of complex acoustic scenes, and can explicit scene structure from specialised detectors make them more reliable?**

Possible directions:

1. **Benchmark:** a larger, controlled scene-structure benchmark (overlap, SNR, reverberation, counts, durations) covering several audio-LLMs.
2. **Grounding:** inject detector output into an audio-LLM (as text or as tokens) and measure the reduction in hallucinations.
3. **Learning:** fine-tune an audio-LLM (e.g. with LoRA) on foreground/background and temporal supervision generated by the synthetic pipeline.
4. **Applications:** explainable monitoring (industrial sounds, urban noise), accessible audio descriptions, and searchable media archives.

## Credits

* PANNs: Kong et al., *PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition*, 2020 ([code](https://github.com/qiuqiangkong/audioset_tagging_cnn), MIT)
* ESC-50: Piczak, *ESC: Dataset for Environmental Sound Classification*, 2015 (CC BY-NC 3.0)
* Qwen2.5 and Qwen2-Audio: Qwen team, Alibaba Cloud (see the model cards for licenses)
