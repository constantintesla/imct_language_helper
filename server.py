"""Локальный переводчик RU -> ZH. Только стандартная библиотека Python.
Перевод делает Ollama (локально). Запуск: python server.py  ->  http://localhost:8000
"""
import json, os, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = os.environ.get("MODEL", "qwen2.5:7b")
OLLAMA = os.environ.get("OLLAMA", "http://localhost:11434")
HERE = os.path.dirname(os.path.abspath(__file__))

SYSTEM = (
    "Ты синхронный переводчик с русского на китайский (упрощённые иероглифы). "
    "Переводи естественно, как говорит носитель, сохраняя смысл и тон. "
    "Учитывай предыдущие фразы как контекст, но переводи ТОЛЬКО последнюю. "
    "Выводи только перевод, без пояснений, без пиньиня, без кавычек."
)


def translate(text, history):
    msgs = [{"role": "system", "content": SYSTEM}]
    for ru, zh in history[-3:]:
        msgs += [{"role": "user", "content": ru}, {"role": "assistant", "content": zh}]
    msgs.append({"role": "user", "content": text})
    body = json.dumps({"model": MODEL, "messages": msgs, "stream": False,
                       "keep_alive": "30m", "options": {"temperature": 0.2}}).encode()
    req = urllib.request.Request(OLLAMA + "/api/chat", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["message"]["content"].strip()


VOICES = {"zh-CN-XiaoxiaoNeural", "zh-CN-XiaoyiNeural", "zh-CN-YunxiNeural", "zh-CN-YunjianNeural"}
_tts_cache = {}


def tts(text, voice):
    import asyncio, edge_tts
    key = (text, voice)
    if key not in _tts_cache:
        async def run():
            out = b""
            async for ch in edge_tts.Communicate(text, voice).stream():
                if ch["type"] == "audio":
                    out += ch["data"]
            return out
        _tts_cache[key] = asyncio.run(run())
    return _tts_cache[key]


class H(BaseHTTPRequestHandler):
    def _send(self, code, data, ctype="application/json"):
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html")
        elif self.path.startswith("/tts?"):
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            voice = q.get("voice", ["zh-CN-XiaoxiaoNeural"])[0]
            try:
                if voice not in VOICES:
                    raise ValueError("bad voice")
                self._send(200, tts(q["text"][0], voice), "audio/mpeg")
            except Exception as e:
                self._send(500, {"error": str(e)})
        elif self.path == "/health":
            try:
                with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=3) as r:
                    names = [m["name"] for m in json.load(r)["models"]]
                self._send(200, {"ollama": True, "model": MODEL, "installed": MODEL in names or any(n.startswith(MODEL) for n in names), "models": names})
            except Exception:
                self._send(200, {"ollama": False, "model": MODEL, "installed": False})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/translate":
            return self._send(404, {"error": "not found"})
        try:
            d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self._send(200, {"zh": translate(d["text"], d.get("history", []))})
        except urllib.error.URLError:
            self._send(503, {"error": "Ollama не запущена (ollama serve)"})
        except Exception as e:
            self._send(500, {"error": str(e)})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"Открой http://localhost:8000   (модель: {MODEL})")
    ThreadingHTTPServer(("127.0.0.1", 8000), H).serve_forever()
