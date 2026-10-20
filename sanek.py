"""Русский диктор «Санек»: F5-TTS (ESpeech, клонирование голоса по образцу) + ударения RUAccent.
Логика перенесена из lms_translater_src (explainer/f5voice.py, player/project.py::synth), без проверки через Whisper и без Silero.
Лицензия весов F5-TTS Russian — CC-BY-NC-4.0 (некоммерческое использование)."""
import json, os, re

import numpy as np

import speech_ru

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(HERE, "models"))
VOICES_DIR = os.path.join(HERE, "voices")
SPEED = float(os.environ.get("SANEK_SPEED", "1.2"))   # 1.2 — как в оригинальном проекте
NFE = int(os.environ.get("SANEK_NFE", "48"))
SR = 24000
VOWELS = "аеёиоуыэюя"
FIX = {"чт+обы": "чтоб+ы", "чт+об": "чтоб", "ч+то": "чт+о", "эт+от": "+этот"}
PAUSE = {".": 0.45, "!": 0.45, "?": 0.45, "…": 0.5, ":": 0.35, ";": 0.35}


def tidy(stressed):
    """RUAccent ставит ударение даже в односложных словах — TTS их «тянет»; убираем и чиним несколько частых слов."""
    def fix(m):
        w = m.group(0)
        bare = w.replace("+", "")
        if sum(c in VOWELS for c in bare.lower()) <= 1 and "+" in w and bare.lower() not in ("что", "это"):
            return bare
        return FIX.get(w.lower(), w)
    return re.sub(r"[\w+]+", fix, stressed)


def trim(y, thr=0.01, keep=0.08):
    loud = np.where(np.abs(y) > thr * max(np.abs(y).max(), 1e-6))[0]
    if not len(loud):
        return y
    return y[max(loud[0] - int(keep * SR), 0): loud[-1] + int(keep * SR)]


class Sanek:
    def __init__(self, voice="sanek"):
        from f5_tts.api import F5TTS
        from ruaccent import RUAccent
        ck = os.path.join(MODELS_DIR, "espeech", "espeech_tts_rlv2.pt")
        vc = os.path.join(MODELS_DIR, "espeech", "vocab.txt")
        if not (os.path.exists(ck) and os.path.exists(vc)):
            raise FileNotFoundError(f"нет весов ESpeech: {ck} (скопируй из lms_translater_src/data/models/espeech)")
        meta = json.load(open(os.path.join(VOICES_DIR, voice + ".json"), encoding="utf-8"))
        self.ref = os.path.join(VOICES_DIR, meta["wav"])
        self.acc = RUAccent()
        self.acc.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False)
        self.ref_text = tidy(self.stress(meta["text"]))   # русский образец ударяется так же, как озвучиваемый текст
        self.f5 = F5TTS(model="F5TTS_v1_Base", ckpt_file=ck, vocab_file=vc, device="cuda")

    def stress(self, text):
        return self.acc.process_all(text.replace("+", ""))

    def say(self, text):
        """Текст -> (float32 моно, 24 кГц). Предложение за предложением, как в оригинале."""
        import torch
        spoken = speech_ru.to_spoken(text).strip()
        sents = [x for x in re.split(r"(?<=[.!?…])\s+", spoken) if x]
        parts = []
        for i, sent in enumerate(sents):
            gen = tidy(self.stress(sent))
            torch.manual_seed(int(np.random.randint(1 << 20)))
            wav, sr, _ = self.f5.infer(ref_file=self.ref, ref_text=self.ref_text, gen_text=gen, nfe_step=NFE,
                                       cfg_strength=2.0, sway_sampling_coef=-1.0, speed=SPEED, remove_silence=False)
            y = trim(np.asarray(wav).squeeze().astype(np.float32))
            y = (y / max(np.abs(y).max(), 1e-3) * 0.9).astype(np.float32)
            parts.append(y)
            if i + 1 < len(sents):
                parts.append(np.zeros(int(PAUSE.get(sent.rstrip()[-1:], 0.3) * SR), np.float32))
        return np.concatenate(parts) if parts else np.zeros(SR // 10, np.float32)
