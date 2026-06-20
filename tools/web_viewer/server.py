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
_lock = threading.Lock()


def _ai_loop() -> None:
    """백그라운드에서 AI를 돌리며 매 프레임 Live2D 파라미터를 갱신한다."""
    global _latest
    s = BroadcastStreamer(seed=2025)
    try:
        model = s.load_avatar(str(MODEL_DIR))
        print(f"[viewer] 모델 인식: {model.summary()}")
        have_model = True
    except Exception as e:
        print(f"[viewer] 모델 로드 실패({e}) — 기본 파라미터로 진행")
        have_model = False

    t0 = time.perf_counter()
    while True:
        t = time.perf_counter() - t0
        # 7초 주기로 약 2.5초간 '말하기'(립싱크 보여주기)
        speaking = (t % 7.0) < 2.5
        s._speaking = speaking
        if speaking:
            # 모음처럼 입을 여닫는 립싱크 진폭
            s._viseme_open = max(0.0, 0.5 + 0.5 * math.sin(t * 13.0)) * 0.9
            s._energy = 0.8
        try:
            frame = s.render_live2d_frame() if have_model else {}
        except Exception:
            frame = {}
        with _lock:
            _latest = {"speaking": speaking, "params": frame}
        time.sleep(1.0 / FPS)


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path.split("?")[0] == "/frame":
            with _lock:
                body = json.dumps(_latest).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
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
