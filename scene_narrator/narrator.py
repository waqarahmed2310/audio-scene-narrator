"""LLM layer: reason over the detector timeline in natural language.

The LLM never hears the audio. It only sees the structured timeline, which makes every
statement traceable to a detection (and makes hallucinations easy to spot).
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are an audio scene analyst. You cannot hear the recording. You only see the output of a sound event detector: detected sound events with time stamps, a role and a confidence score.
- FOREGROUND = a salient sound that stands out from the scene (louder while active, usually short).
- BACKGROUND = an ambient sound that is present for a long time at a steady level.
Rules:
1. Base every statement only on the detector output. Never mention sounds that are not listed.
2. Labels come from the AudioSet ontology and can be redundant (e.g. "Dog" and "Bark" describe the same sound). Treat such labels as one sound.
3. Events with confidence below 0.3 are uncertain; say so if you mention them.
4. If the detector output does not contain the information needed, say that the detector did not find it.
5. Refer to times in seconds."""

DESCRIBE_PROMPT = """Detector output:
{timeline}

Write a short, natural description (3-5 sentences) of this acoustic scene for a listener. Mention the background ambience first, then the foreground events in time order, and finish with a guess of the likely setting."""

ANSWER_PROMPT = """Detector output:
{timeline}

Question: {question}
Answer concisely. If the question asks for a single word or a yes/no answer, start your reply with that word."""


class HFChatLLM:
    """Local instruction-tuned model from the Hugging Face Hub (fits a free Colab T4)."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-3B-Instruct", load_in_4bit: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        kwargs = {"device_map": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        try:  # newer transformers use `dtype`, older ones `torch_dtype`
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, **kwargs)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, **kwargs)
        self.model.eval()

    def chat(self, messages: list[dict], max_new_tokens: int = 256) -> str:
        import torch

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class OpenAICompatibleLLM:
    """Any OpenAI-compatible chat endpoint (OpenAI, Groq, a local vLLM/Ollama server, ...)."""

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1", api_key: str | None = None):
        import os
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def chat(self, messages: list[dict], max_new_tokens: int = 256) -> str:
        import requests
        r = requests.post(f"{self.base_url}/chat/completions",
                          headers={"Authorization": f"Bearer {self.api_key}"},
                          json={"model": self.model, "messages": messages,
                                "max_tokens": max_new_tokens + 1024, "temperature": 0},
                          timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


class Narrator:
    def __init__(self, llm):
        self.llm = llm

    def describe(self, timeline: str) -> str:
        return self.llm.chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": DESCRIBE_PROMPT.format(timeline=timeline)},
        ], max_new_tokens=220)

    def answer(self, timeline: str, question: str, history: list[tuple[str, str]] | None = None) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for q, a in (history or [])[-4:]:  # short memory for follow-up questions
            messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        messages.append({"role": "user", "content": ANSWER_PROMPT.format(timeline=timeline, question=question)})
        return self.llm.chat(messages, max_new_tokens=160)
