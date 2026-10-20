"""Локальный переводчик RU -> ZH. Всё работает на этом ПК, без интернета (после первой загрузки моделей):
  речь -> текст : faster-whisper (GPU)
  перевод       : LM Studio (OpenAI-совместимый API)
  озвучка       : Kokoro-82M (GPU)
Запуск: python server.py  -> https://<IP-этого-ПК>:8443 (HTTPS нужен браузеру для доступа к микрофону по сети)
"""
import asyncio, glob, io, os, re, site, threading, time, wave

for _p in glob.glob(os.path.join(site.getsitepackages()[-1], "nvidia", "*", "bin")):  # CUDA-библиотеки из pip
    os.add_dll_directory(_p)
    os.environ["PATH"] = _p + os.pathsep + os.environ["PATH"]
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
if os.environ.get("OFFLINE") == "1":
    os.environ["HF_HUB_OFFLINE"] = "1"

import httpx, numpy as np, uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from certs import ensure_cert, lan_ips

LMSTUDIO = os.environ.get("LMSTUDIO", "http://localhost:1234/v1")
MODEL = os.environ.get("MODEL", "qwen/qwen3.5-35b-a3b")
WHISPER = os.environ.get("WHISPER", "large-v3-turbo")        # русский
WHISPER_ZH = os.environ.get("WHISPER_ZH", "large-v3")         # китайский: полная модель заметно точнее turbo
PORT = int(os.environ.get("PORT", "8443"))
HERE = os.path.dirname(os.path.abspath(__file__))

# направления перевода: ru2zh (русский -> китайский) и zh2ru (китайский -> русский)
SYSTEMS = {
    "ru2zh": (
        "Ты синхронный переводчик с русского на китайский (упрощённые иероглифы). "
        "Переводи естественно, как говорит носитель, сохраняя смысл и тон. "
        "Учитывай предыдущие фразы как контекст, но переводи ТОЛЬКО последнюю. "
        "Выводи только перевод, без пояснений, без пиньинь, без кавычек."),
    "zh2ru": (
        "Ты синхронный переводчик с китайского на русский. "
        "Переводи естественно, как говорит носитель, сохраняя смысл и тон. "
        "Учитывай предыдущие фразы как контекст, но переводи ТОЛЬКО последнюю. "
        "Выводи только перевод на русском, без пояснений, без пиньинь, без кавычек. "
        "Не добавляй комментарии и пометки в скобках; бессмысленные повторы слов переводи один раз."),
}
SRC_LANG = {"ru2zh": "ru", "zh2ru": "zh"}  # язык речи, который слушаем
DST_LANG = {"ru2zh": "zh", "zh2ru": "ru"}  # язык озвучки перевода
WHISPER_PROMPT = {"ru": "Обычный разговор на русском языке.", "zh": "以下是普通话的句子，使用简体中文。"}

VOICES = {
    "zh": {
        "zf_xiaobei": "Xiaobei (жен.)", "zf_xiaoni": "Xiaoni (жен.)", "zf_xiaoxiao": "Xiaoxiao (жен.)",
        "zf_xiaoyi": "Xiaoyi (жен.)", "zm_yunxi": "Yunxi (муж.)", "zm_yunjian": "Yunjian (муж.)",
        "zm_yunxia": "Yunxia (муж.)", "zm_yunyang": "Yunyang (муж.)",
    },
    "ru": {"sanek": "Санек (муж.)"},
}
# типичные галлюцинации Whisper на тишине/шуме
JUNK = ("продолжение следует", "субтитры", "спасибо за просмотр", "редактор субтитров", "подписывайтесь",
        "字幕", "请不吝点赞", "谢谢观看", "订阅")

# зацикливание Whisper: «我会带你,我会带你,我会带你…» -> одна фраза
_REPEAT = re.compile(r"(.{2,40}?)(?:[\s,，、.。!！?？;；-]*\1){2,}", re.S)


def collapse_repeats(text):
    prev = None
    while prev != text:
        prev, text = text, _REPEAT.sub(r"\1", text)
    return text


app = FastAPI()
_stt = {}
_tts = _tts_ru = None
_stt_lock, _tts_lock, _load_lock = threading.Lock(), threading.Lock(), threading.Lock()
_tts_cache = {}
CA_PATH = None


def get_stt(lang="ru"):
    name = WHISPER_ZH if lang == "zh" else WHISPER
    with _load_lock:
        if name not in _stt:
            from faster_whisper import WhisperModel
            _stt[name] = WhisperModel(name, device="cuda", compute_type="float16")
    return _stt[name]


def get_tts():
    global _tts
    with _load_lock:
        if _tts is None:
            from kokoro import KPipeline
            _tts = KPipeline(lang_code="z", repo_id="hexgrad/Kokoro-82M", device="cuda")
    return _tts


def get_tts_ru():
    global _tts_ru
    with _load_lock:
        if _tts_ru is None:
            from sanek import Sanek
            _tts_ru = Sanek()
    return _tts_ru


def wav_bytes(samples, rate):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "index.html"), headers={"Cache-Control": "no-store"})


@app.get("/ca.crt")
def ca_cert():
    return FileResponse(CA_PATH, media_type="application/x-x509-ca-cert", filename="imct-translator-ca.crt")


@app.post("/stt")
async def stt(request: Request, dir: str = "ru2zh"):
    lang = SRC_LANG.get(dir, "ru")
    pcm = np.frombuffer(await request.body(), dtype=np.int16).astype(np.float32) / 32768  # 16 кГц, моно
    if len(pcm) < 4800:
        return {"text": ""}

    def run():
        with _stt_lock:
            model = get_stt(lang)
            # язык по звуку: чужую речь (в т.ч. собственную озвучку из динамиков) отбрасываем
            probs = dict(model.detect_language(pcm, vad_filter=True)[2])
            p_src, p_other = probs.get(lang, 0.0), probs.get("zh" if lang == "ru" else "ru", 0.0)
            if p_src < p_other:
                print(f"STT[{lang}] пропущено: язык {lang}={p_src:.2f} < другой={p_other:.2f}", flush=True)
                return ""
            segs, _ = model.transcribe(
                pcm, language=lang, beam_size=5, vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 300},
                temperature=[0.0, 0.2, 0.4, 0.6], compression_ratio_threshold=2.0, log_prob_threshold=-1.0,
                repetition_penalty=1.15, no_repeat_ngram_size=4,
                condition_on_previous_text=False, no_speech_threshold=0.6,
                initial_prompt=WHISPER_PROMPT[lang])
            return " ".join(s.text.strip() for s in segs).strip()

    text = collapse_repeats(await asyncio.to_thread(run))
    if any(j in text.lower() for j in JUNK):
        text = ""
    print(f"STT[{lang}] {len(pcm) / 16000:.1f}s: {text!r}", flush=True)
    return {"text": text}


@app.post("/translate")
async def translate(request: Request):
    d = await request.json()
    msgs = [{"role": "system", "content": SYSTEMS.get(d.get("dir"), SYSTEMS["ru2zh"])}]
    for src, dst in d.get("history", [])[-3:]:
        msgs += [{"role": "user", "content": src}, {"role": "assistant", "content": dst}]
    msgs.append({"role": "user", "content": d["text"]})
    try:
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(LMSTUDIO + "/chat/completions", json={
                "model": MODEL, "messages": msgs, "temperature": 0.2, "max_tokens": 600,
                "reasoning_effort": "none"})  # отключает «размышления» у Qwen/Gemma в LM Studio
        r.raise_for_status()
        out = (r.json()["choices"][0]["message"]["content"] or "").strip()
    except httpx.ConnectError:
        return JSONResponse({"error": "LM Studio не запущен (Developer → Start Server)"}, 503)
    except Exception as e:
        return JSONResponse({"error": f"LM Studio: {e}"}, 500)
    if not out:
        return JSONResponse({"error": "пустой ответ модели"}, 502)
    return {"text": out}


@app.get("/tts")
async def tts(text: str, voice: str, lang: str = "zh"):
    if voice not in VOICES.get(lang, {}):
        raise HTTPException(400, "bad voice")
    key = (text, voice, lang)

    def run():
        if key not in _tts_cache:
            with _tts_lock:
                if lang == "ru":
                    _tts_cache[key] = wav_bytes(get_tts_ru().say(text), 24000)
                else:
                    parts = [a.cpu().numpy() for _, _, a in get_tts()(text, voice=voice, speed=1.0)]
                    _tts_cache[key] = wav_bytes(np.concatenate(parts), 24000)
            if len(_tts_cache) > 200:
                _tts_cache.pop(next(iter(_tts_cache)))
        return _tts_cache[key]

    return Response(await asyncio.to_thread(run), media_type="audio/wav")


@app.get("/health")
async def health():
    out = {"model": MODEL, "voices": VOICES, "stt": WHISPER, "stt_zh": WHISPER_ZH, "dst": DST_LANG}
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            ids = [m["id"] for m in (await c.get(LMSTUDIO + "/models")).json()["data"]]
        out.update(llm=True, installed=MODEL in ids, models=ids)
    except Exception:
        out.update(llm=False, installed=False)
    return out


if __name__ == "__main__":
    crt, key, CA_PATH = ensure_cert(os.path.join(HERE, "certs"))
    print("Прогрев моделей (первый запуск скачивает веса)...")
    t = time.time()
    get_tts(); get_tts_ru()
    for lg in ("ru", "zh"):
        get_stt(lg).transcribe(np.zeros(16000, dtype=np.float32), language=lg)
    print(f"Готово за {time.time() - t:.0f} с. LLM: {MODEL} @ {LMSTUDIO}")
    print(f"Открой: https://localhost:{PORT}")
    for ip in lan_ips():
        print(f"        https://{ip}:{PORT}   ")
    print(f"Сертификат для устройств (поставить один раз): https://<IP>:{PORT}/ca.crt  или {CA_PATH}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, ssl_certfile=crt, ssl_keyfile=key, log_level="warning")
