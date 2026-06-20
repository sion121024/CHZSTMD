"""VTube Studio 플러그인 — 우리 AI가 VTS 안의 youling을 실시간 조종.

VTS Public API(WebSocket, 기본 ws://localhost:8001)에 붙어, 우리 단일 신경망이
만든 모션/립싱크/감정을 VTS 기본 트래킹 파라미터(FaceAngleX, MouthOpen, ...)에
주입한다. youling이 VTS에서 그대로 움직인다(물리·표정·렌더는 VTS가 담당).

준비
----
1) VTube Studio 실행 → 설정에서 "Start API (allow plugins)" 켜기 (포트 8001).
2) `pip install websocket-client`
3) `python tools/vtube_studio/vts_driver.py`
4) 처음 실행 시 VTS에 플러그인 허용 팝업 → "Allow" 클릭(토큰은 vts_token.txt에 저장).

감정 표정은 VTS의 '핫키(표정)'로 연결한다 — 핫키 이름에 happy/smile/angry/surprise
등이 들어있으면 감정에 맞춰 자동 트리거한다(없으면 파라미터로만 표현).
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

try:
    from websocket import create_connection  # websocket-client
except Exception:
    print("websocket-client 가 필요합니다:  pip install websocket-client")
    raise SystemExit(1)

from broadcast_ai.runtime import BroadcastStreamer  # noqa: E402

VTS_URL = "ws://localhost:8001"
PLUGIN_NAME = "CHZSTMD AI Driver"
PLUGIN_DEV = "CHZSTMD"
TOKEN_FILE = Path(__file__).resolve().parent / "vts_token.txt"
FPS = 30

# 감정 → 표정 핫키 이름 키워드(있으면 트리거)
EMO_HOTKEY_KW = {
    "excited": ["excited", "happy", "joy", "smile", "기쁨", "웃"],
    "amused": ["smile", "happy", "laugh", "fun", "웃", "즐"],
    "surprised": ["surprise", "shock", "wow", "놀"],
    "frustrated": ["angry", "mad", "sad", "annoy", "화", "짜증", "슬"],
    "focused": ["serious", "focus", "calm", "진지", "집중"],
    "neutral": ["neutral", "default", "idle", "기본"],
}


class VTS:
    def __init__(self, url=VTS_URL):
        self.ws = create_connection(url, timeout=10)
        self._rid = 0

    def _req(self, mtype, data=None):
        self._rid += 1
        msg = {"apiName": "VTubeStudioPublicAPI", "apiVersion": "1.0",
               "requestID": f"r{self._rid}", "messageType": mtype, "data": data or {}}
        self.ws.send(json.dumps(msg))
        return json.loads(self.ws.recv())

    def authenticate(self):
        token = TOKEN_FILE.read_text().strip() if TOKEN_FILE.exists() else None
        if not token:
            print("VTS에서 플러그인 허용 팝업이 뜨면 'Allow'를 누르세요…")
            r = self._req("AuthenticationTokenRequest",
                          {"pluginName": PLUGIN_NAME, "pluginDeveloper": PLUGIN_DEV})
            token = r.get("data", {}).get("authenticationToken")
            if not token:
                raise RuntimeError(f"토큰 발급 실패: {r}")
            TOKEN_FILE.write_text(token)
        r = self._req("AuthenticationRequest",
                      {"pluginName": PLUGIN_NAME, "pluginDeveloper": PLUGIN_DEV,
                       "authenticationToken": token})
        if not r.get("data", {}).get("authenticated"):
            # 토큰 만료 등 → 토큰 파일 지우고 한 번 더
            TOKEN_FILE.unlink(missing_ok=True)
            raise RuntimeError(f"인증 실패: {r.get('data')}")
        print("VTS 인증 성공.")

    def inject(self, params: dict[str, float]):
        self._req("InjectParameterDataRequest", {
            "faceFound": True, "mode": "set",
            "parameterValues": [{"id": k, "value": v} for k, v in params.items()],
        })

    def hotkeys(self) -> list[dict]:
        r = self._req("HotkeysInCurrentModelRequest", {})
        return r.get("data", {}).get("availableHotkeys", [])

    def trigger_hotkey(self, hotkey_id: str):
        self._req("HotkeyTriggerRequest", {"hotkeyID": hotkey_id})


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def map_to_vts(frame: dict) -> dict[str, float]:
    """우리 rig 프레임 → VTS 기본 트래킹 파라미터."""
    head = frame["head"]            # [yaw, pitch, roll]
    mouth = frame["mouth"]
    eyes = frame["eyes"]
    return {
        "FaceAngleX": _clamp(head[0] * 110.0, -30, 30),
        "FaceAngleY": _clamp(head[1] * 110.0, -30, 30),
        "FaceAngleZ": _clamp(head[2] * 110.0, -30, 30),
        "FacePositionX": _clamp(frame["body_sway"] * 3.0, -1, 1),
        "MouthOpen": _clamp(mouth["open"], 0, 1),
        "MouthSmile": _clamp(0.5 + 0.5 * mouth["wide"] + mouth["smile"], 0, 1),
        "EyeOpenLeft": _clamp(1.0 - eyes["blink"], 0, 1),
        "EyeOpenRight": _clamp(1.0 - eyes["blink"], 0, 1),
        "Brows": _clamp(eyes["brow"], -1, 1),
    }


def _talk_mouth(t):
    syl = t * 5.5
    i = int(syl); frac = syl - i
    r = (math.sin(i * 12.9898) * 43758.5453) % 1.0
    return max(0.0, (0.3 + 0.6 * r) * (math.sin(math.pi * frac) ** 1.4)), \
        0.35 * ((math.sin(i * 7.13) + 1) / 2) - 0.1


EMO_CYCLE = ["amused", "neutral", "excited", "focused", "surprised", "excited", "neutral"]


def main():
    vts = VTS()
    vts.authenticate()

    # 감정 → 핫키 매핑
    hks = vts.hotkeys()
    print(f"VTS 핫키 {len(hks)}개:", [h.get("name") for h in hks][:12])
    emo2hk = {}
    for emo, kws in EMO_HOTKEY_KW.items():
        for h in hks:
            name = (h.get("name") or "").lower()
            if any(k.lower() in name for k in kws):
                emo2hk[emo] = h.get("hotkeyID"); break

    s = BroadcastStreamer(seed=2025)
    last_emo = None
    t0 = time.perf_counter()
    print("AI 구동 시작 — VTS의 youling이 움직입니다. (Ctrl+C 종료)")
    try:
        while True:
            t = time.perf_counter() - t0
            speaking = (t % 8.0) < 3.0
            s._speaking = speaking
            s.emotion_override = "excited" if speaking else EMO_CYCLE[int(t / 5.0) % len(EMO_CYCLE)]
            if speaking:
                s._viseme_open, s._viseme_wide = _talk_mouth(t)
                s._energy = 0.8
            else:
                s._viseme_open = s._viseme_wide = 0.0

            frame = s.render_frame()                 # rig 프레임(감정·모션·립싱크 반영)
            vts.inject(map_to_vts(frame))

            # 감정 바뀌면 표정 핫키 트리거
            if s._emotion != last_emo and s._emotion in emo2hk:
                vts.trigger_hotkey(emo2hk[s._emotion])
                last_emo = s._emotion

            time.sleep(1.0 / FPS)
    except KeyboardInterrupt:
        print("\n종료.")


if __name__ == "__main__":
    main()
