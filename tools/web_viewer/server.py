"""Live2D 웹 뷰어 서버 — 우리 AI가 youling을 실시간 구동.

표준 라이브러리만 사용(추가 설치 불필요). 한 포트에서:
  * 정적 파일(뷰어 HTML + 모델 파일)을 서빙하고
  * /frame 엔드포인트로 AI가 매 프레임 만든 Live2D 파라미터(JSON)를 제공한다.

브라우저 뷰어가 /frame 을 30fps로 받아 모델 파라미터에 적용 → youling이 우리
단일 신경망이 생성한 호흡/깜빡임/표정/립싱크대로 움직인다.

실행:  python tools/web_viewer/server.py
브라우저:  http://localhost:8765/tools/web_viewer/index.html
"""

from __future__ import annotations

import functools
import json
import math
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from broadcast_ai.runtime import BroadcastStreamer  # noqa: E402

PORT = 8765
FPS = 45
MODEL_DIR = ROOT / "assets" / "avatar" / "gothic_lolita"

_latest: dict = {}
_model = None          # Live2DModel (표정/파트/표식 조회용)
_streamer = None       # BroadcastStreamer (제스처 트리거용)
_lock = threading.Lock()

# 감정 무드 순환(데모): ~5초마다 바뀌어 표정이 감정 따라 변하는 걸 보여준다.
EMO_CYCLE = ["amused", "neutral", "excited", "focused", "amused", "surprised", "excited", "neutral"]


def _talk_mouth(t: float):
    """음절 기반 입 움직임 — 단순 사인보다 말하는 것처럼 자연스럽게.

    음절마다(약 5.5Hz) 진폭이 다른 입벌림 + 모음별 입너비 변화.
    """
    syl = t * 5.5
    i = int(syl)
    frac = syl - i
    r = (math.sin(i * 12.9898) * 43758.5453) % 1.0   # 음절별 의사난수 진폭
    peak = 0.3 + 0.6 * r
    openv = max(0.0, peak * (math.sin(math.pi * frac) ** 1.4))   # 음절 사이엔 닫힘
    wide = 0.35 * ((math.sin(i * 7.13) + 1) / 2) - 0.1
    return openv, wide


def _parse_exp3(path) -> list[dict]:
    """exp3.json → [{id, value, blend}] 파라미터 오버라이드 목록."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return []
    out = []
    for p in data.get("Parameters", []):
        out.append({"id": p.get("Id"), "value": p.get("Value", 0.0),
                    "blend": p.get("Blend", "Overwrite")})
    return out


def _expressions_payload() -> list[dict]:
    if _model is None:
        return []
    return [{"name": name, "params": _parse_exp3(path)}
            for name, path in _model.expressions]


def _inspect_payload() -> dict:
    if _model is None:
        return {"parameters": [], "parts": [], "watermark": []}
    wm = _model.watermark_candidates()
    return {
        "parameters": [{"id": i, "name": _model.parameter_names.get(i, i)}
                       for i in _model.parameter_ids],
        "parts": [{"id": i, "name": _model.part_names.get(i, i)}
                  for i in _model.part_ids],
        "watermark": [{"kind": k, "id": i, "name": n} for k, i, n in wm],
        "watermark_part_ids": [i for k, i, n in wm if k == "part"],
        "watermark_param_ids": [i for k, i, n in wm if k == "param"],
    }


def _ai_loop() -> None:
    """백그라운드에서 AI를 돌리며 매 프레임 Live2D 파라미터를 갱신한다."""
    global _latest, _model, _streamer
    s = BroadcastStreamer(seed=2025)
    _streamer = s
    try:
        model = s.load_avatar(str(MODEL_DIR))
        _model = model
        print(f"[viewer] 모델 인식: {model.summary()}")
        print(f"[viewer] 표정 {len(model.expressions)}개 · 표식 후보 {model.watermark_candidates()}")
        have_model = True
    except Exception as e:
        print(f"[viewer] 모델 로드 실패({e}) — 기본 파라미터로 진행")
        have_model = False

    t0 = time.perf_counter()
    while True:
        t = time.perf_counter() - t0
        # 8초 주기로 약 3초간 '말하기'(립싱크 시연)
        speaking = (t % 8.0) < 3.0
        s._speaking = speaking
        # 감정 무드: ~5초마다 바뀌고 말할 땐 들뜸 → 표정이 감정 따라 변함
        s.emotion_override = "excited" if speaking else EMO_CYCLE[int(t / 5.0) % len(EMO_CYCLE)]
        if speaking:
            openv, wide = _talk_mouth(t)
            s._viseme_open, s._viseme_wide = openv, wide
            s._energy = 0.8
        else:
            s._viseme_open = s._viseme_wide = 0.0
        try:
            frame = s.render_live2d_frame() if have_model else {}
        except Exception:
            frame = {}
        with _lock:
            _latest = {"speaking": speaking, "emotion": s.emotion_override, "params": frame}
        time.sleep(1.0 / FPS)


class Handler(SimpleHTTPRequestHandler):
    def _json(self, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        route = self.path.split("?")[0]
        if route == "/frame":
            with _lock:
                return self._json(_latest)
        if route == "/expressions":
            return self._json(_expressions_payload())
        if route == "/inspect":
            return self._json(_inspect_payload())
        if route == "/gesture":
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            name = (q.get("name") or [""])[0]
            ok = bool(_streamer and _streamer.trigger_gesture(name))
            return self._json({"ok": ok, "gesture": name})
        if route == "/command":
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            text = (q.get("text") or [""])[0]
            did = _streamer.command(text) if _streamer else None
            return self._json({"ok": did is not None, "gesture": did})
        return super().do_GET()

    def log_message(self, *args):  # 조용히
        pass


def main() -> None:
    threading.Thread(target=_ai_loop, daemon=True).start()
    handler = functools.partial(Handler, directory=str(ROOT))
    httpd = ThreadingHTTPServer(("127.0.0.1", PORT), handler)
    url = f"http://localhost:{PORT}/tools/web_viewer/index.html"
    print("=" * 60)
    print(" Live2D 웹 뷰어 실행 중")
    print(f"  브라우저에서 열기:  {url}")
    print("  종료: Ctrl+C")
    print("=" * 60)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n종료.")


if __name__ == "__main__":
    main()
