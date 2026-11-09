"""End-to-end baseline: an audio-language model that listens to the audio directly.

Default: Qwen2-Audio-7B-Instruct in 4-bit (about 6 GB of GPU memory, fits a Colab T4).
Free the cascade's LLM before loading this one.
"""

from __future__ import annotations

import librosa


class Qwen2AudioQA:
    def __init__(self, model_id: str = "Qwen/Qwen2-Audio-7B-Instruct", load_in_4bit: bool = True):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2AudioForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_id)
        kwargs = {"device_map": "auto"}
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        try:
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                model_id, dtype=torch.float16, **kwargs)
        except TypeError:
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.float16, **kwargs)
        self.model.eval()
        self.sr = self.processor.feature_extractor.sampling_rate

    def ask(self, audio_path: str, question: str, max_new_tokens: int = 64) -> str:
        import torch

        conversation = [{"role": "user", "content": [
            {"type": "audio", "audio_url": audio_path},
            {"type": "text", "text": question},
        ]}]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audio = librosa.load(audio_path, sr=self.sr)[0]
        try:  # the keyword was renamed from `audios` to `audio` in newer transformers
            inputs = self.processor(text=text, audio=[audio], sampling_rate=self.sr,
                                    return_tensors="pt", padding=True)
        except (TypeError, ValueError):
            inputs = self.processor(text=text, audios=[audio], sampling_rate=self.sr,
                                    return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device)
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(self.model.dtype)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        out = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()
