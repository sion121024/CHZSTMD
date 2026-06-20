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


# 머리/몸 게인 — 우리 모션 헤드 출력은 ±0.05 수준으로 작다(EMA 평활 후 더 작음).
# VTS는 '각도(도)'를 받으므로 그대로 넣으면 거의 안 움직인다. 보이는 범위로 키운다.
# (사인파 합성이 아니라 'AI가 만든 신호'를 증폭하는 것 — 고정 애니메이션 아님)
HEAD_GAIN = 185.0      # 평상시 ~5도, 큰 동작은 ~28도(클램프)
SWAY_GAIN = 9.0


def map_to_vts(frame: dict) -> dict[str, float]:
    """우리 rig 프레임 → VTS 기본 트래킹 파라미터(보이는 범위로 증폭)."""
    head = frame["head"]            # [yaw, pitch, roll]
    mouth = frame["mouth"]
    eyes = frame["eyes"]
    breath = frame.get("breath", 0.0)
    sway = frame["body_sway"]
    return {
        # 머리 — AI 모션을 보이는 각도로. 좌우(X)는 몸통 흔들림도 약간 더해 생동감.
        "FaceAngleX": _clamp(head[0] * HEAD_GAIN + sway * 6.0, -28, 28),
        "FaceAngleY": _clamp(head[1] * HEAD_GAIN, -25, 25),
        "FaceAngleZ": _clamp(head[2] * HEAD_GAIN * 0.9, -28, 28),
        # 몸 위치 — 좌우 무게중심 + 호흡에 따른 미세한 상하. 모델이 '살아있게' 보임.
        "FacePositionX": _clamp(sway * SWAY_GAIN, -10, 10),
        "FacePositionY": _clamp((breath - 0.5) * 2.2, -10, 10),
        "MouthOpen": _clamp(mouth["open"], 0, 1),
        "MouthSmile": _clamp(0.5 + 0.5 * mouth["wide"] + mouth["smile"], 0, 1),
        # 입모양(아/이): VTS의 MouthForm/MouthX 가 있으면 모음 폭이 더 자연스러움.
        "MouthForm": _clamp(mouth["wide"], -1, 1),
        # 좌우 눈 따로 — 윙크 제스처가 VTS에서도 보인다.
        "EyeOpenLeft": _clamp(eyes.get("open_l", 1.0 - eyes["blink"]), 0, 1),
        "EyeOpenRight": _clamp(eyes.get("open_r", 1.0 - eyes["blink"]), 0, 1),
        "Brows": _clamp(eyes["brow"], -1, 1),
    }


class _Mouth:
    """말할 때 입을 자연스럽게 — 음절 펄스 + 어택/릴리스 평활.

    이전엔 음절 사이에 입이 0으로 '딱' 닫혀 덜덜거렸다(어색함의 원인).
    여기선 약한 베이스 개구 + 부드러운 펄스를 attack/release로 이어 붙인다.
    """

    def __init__(self):
        self.open = 0.0
        self.form = 0.0

    def update(self, t: float, dt: float) -> tuple[float, float]:
        syl = t * 4.6                      # 음절 속도(조금 낮춰 또박또박)
        i = int(syl); frac = syl - i
        r = (math.sin(i * 12.9898) * 43758.5453) % 1.0
        pulse = (0.35 + 0.55 * r) * (math.sin(math.pi * frac) ** 1.1)
        target = 0.12 + 0.9 * max(0.0, pulse)        # 베이스 0.12 → 완전히 안 닫힘
        # 입은 빨리 열리고(어택) 천천히 닫힌다(릴리스) → 더 사람처럼
        k = (1 - math.exp(-dt / 0.045)) if target > self.open else (1 - math.exp(-dt / 0.09))
        self.open += (target - self.open) * k
        # 모음 폭(아↔이) — 음절마다 살짝, 부드럽게
        fwide = 0.45 * ((math.sin(i * 7.13) + 1) / 2) - 0.15
        self.form += (fwide - self.form) * (1 - math.exp(-dt / 0.07))
        return self.open, self.form

    def relax(self, dt: float):
        self.open += (0.0 - self.open) * (1 - math.exp(-dt / 0.12))
        self.form += (0.0 - self.form) * (1 - math.exp(-dt / 0.12))
        return self.open, self.form


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
    mouth = _Mouth()
    last_emo = None
    smooth: dict[str, float] = {}        # 주입값 프레임간 평활 → 끊김 없이 부드럽게

    # 콘솔에 명령을 치면 제스처 재생 — 예: "윙크해줘", "손 흔들어", "wink".
    import threading

    def _stdin_commands():
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            did = s.command(line) or (line if s.trigger_gesture(line) else None)
            print(f"  → 제스처: {did}" if did else "  (인식된 제스처 없음)")
    threading.Thread(target=_stdin_commands, daemon=True).start()

    t0 = time.perf_counter()
    tprev = t0
    print("AI 구동 시작 — VTS의 youling이 움직입니다. (Ctrl+C 종료)")
    print("머리가 안 움직이면 VTS에서 웹캠 트래킹을 끄세요"
          "(설정 → 카메라 Off). 그래야 우리 AI 주입값이 머리를 구동합니다.")
    print("제스처: 이 창에 '윙크해줘' / '손 흔들어' / 'wink' 처럼 입력하면 재생됩니다.")
    try:
        while True:
            now = time.perf_counter()
            t = now - t0
            dt = min(0.1, now - tprev); tprev = now
            speaking = (t % 8.0) < 3.0
            s._speaking = speaking
            s.emotion_override = "excited" if speaking else EMO_CYCLE[int(t / 5.0) % len(EMO_CYCLE)]
            if speaking:
                s._viseme_open, s._viseme_wide = mouth.update(t, dt)
                s._energy = 0.85
            else:
                s._viseme_open, s._viseme_wide = mouth.relax(dt)

            frame = s.render_frame()                 # rig 프레임(감정·모션·립싱크 반영)
            params = map_to_vts(frame)
            # 프레임간 EMA — 머리/몸은 부드럽게, 입/눈은 빠르게 따라가게(반응성 유지)
            for k, v in params.items():
                a = 0.45 if k.startswith(("Mouth", "EyeOpen")) else 0.25
                v = a * v + (1 - a) * smooth.get(k, v)
                smooth[k] = v
                params[k] = v
            vts.inject(params)

            # 감정 바뀌면 표정 핫키 트리거
            if s._emotion != last_emo and s._emotion in emo2hk:
                vts.trigger_hotkey(emo2hk[s._emotion])
                last_emo = s._emotion

            time.sleep(1.0 / FPS)
    except KeyboardInterrupt:
        print("\n종료.")


if __name__ == "__main__":
    main()
