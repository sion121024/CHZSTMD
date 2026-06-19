# 방송용 AI 버추얼 스트리머 (CHZSTMD)

**처음부터 설계한 단 하나의 신경망**이 FPS 게임을 실시간으로 플레이하고, 아바타를
매 프레임 AI로 움직이며, 시청자 채팅에 저지연으로 반응하고, 게임 튜토리얼을 화면만
보고 스스로 배운다. **완전 로컬, 외부 API/사전학습 모델 없음.**

> 설계 제약 (요구사항 그대로)
> - **기존 모델 안 씀 / 처음부터 설계** — torch·transformers·llama.cpp 같은 프레임워크나
>   사전학습 체크포인트를 적재하지 않는다. 선형층·GRU 셀·활성함수를 **순수 파이썬으로
>   직접 구현**(`src/broadcast_ai/nn/`)한 단일 신경망(`model/unified.py`).
> - **AI가 움직임 / 고정 애니메이션 아님** — 아바타 모션은 사인파 클립이 아니라 매
>   프레임 네트워크 forward로 *생성*된다. 비반복적이고 감정·게임 맥락에 반응한다.
> - **FPS급 실시간** — 한 틱 추론이 수백 µs. CPU 순수 파이썬으로도 **1,000Hz 이상**
>   (일반 FPS 60~240Hz의 수 배). 조준은 온라인으로 **실시간 학습**한다.
> - **모델 1개** — 제어(FPS) · 모션(아바타) · 발화의도가 *같은* 은닉 상태에서 나온다.
>   작업별로 모델을 나누지 않는다.

코어와 신경망 모두 **외부 의존성 0**(표준 라이브러리만)으로 지금 바로 실행/테스트된다.

---

## 빠른 시작

**Linux / macOS**
```bash
python3 demo/run_demo.py                 # 엔드투엔드 데모 (PYTHONPATH 불필요 — 스크립트가 src 자동 추가)
pip install pytest && python3 -m pytest  # 테스트 35개 (conftest가 src 자동 추가)
PYTHONPATH=src python3 -m broadcast_ai.runtime.streamer  # 대화형 CLI
```

**Windows (CMD/PowerShell, 예: 갤럭시 북)** — `python3`이 아니라 `python`, `PYTHONPATH=...` 인라인 문법은 쓰지 않는다:
```bat
python demo\run_demo.py        REM 데모 (또는 더블클릭: run_demo.bat)
python -m pytest               REM 테스트 (또는: run_tests.bat)

REM 대화형 CLI는 둘 중 하나:
set PYTHONPATH=src&& python -m broadcast_ai.runtime.streamer
REM 또는 설치 후 어디서나:  pip install -e .   다음   broadcast-ai
```
> ⚠️ `PYTHONPATH=src python3 ...`(리눅스 문법)을 Windows에 그대로 치면
> `'PYTHONPATH'은(는) ... 아닙니다` 오류가 납니다. 위 Windows 방식으로 실행하세요.

데모 출력(발췌):

```
단일모델: from-scratch 신경망, 파라미터 16,636개 (사전학습/외부 모델 없음)

▶ FPS 게임 — 실시간 제어 + 조준 온라인 학습
추론 처리량 : 1,225 Hz  → FPS 실시간 가능: 예 ✅ (일반 FPS 60~240Hz)
조준 오차   : 초반 0.201 → 후반 0.146  (학습으로 개선 ✅)
명중        : 41회

▶ 아바타 — AI가 매 프레임 모션을 *생성* (고정 클립 아님)
  f0: head=[0.062, 0.060, 0.011] body_sway=-0.022 호흡=0.23 (매 프레임 다름)
```

---

## 단 하나의 신경망 (`model/unified.py`)

```
관측 Observation(24) ─▶ 입력 임베딩(Linear) ─▶ GRU 코어(은닉 h, 48) ─┐
                                                                      │ (공유 트렁크)
        ┌──────────────────────┬──────────────────────┬──────────────┘
        ▼                      ▼                      ▼
  제어 헤드               모션 헤드               발화의도 헤드
  aim(2)/move(2)/        12 포즈채널 속도        감정(6)/에너지/말하기
  버튼(4)  + 온라인        → 적분해 살아있는        ↑ 같은 h_t에서 분기
  조준 학습(NLMS)         움직임 생성              (= 모델 1개)
```

- **트렁크**(`embed` + `GRUCell`)와 모든 헤드는 직접 구현한 가중치(Xavier 초기화, 시드
  고정). 사전학습 없음 → 재현 가능하고 그대로 **학습 가능**(RL/모방학습 훅 자리).
- **모션은 네트워크 출력**: 모션 헤드가 포즈 *속도*를 내고, 내부 OU 잡음 드라이브 +
  GRU 동역학 + 감정/에너지가 섞여 비반복적·유기적 움직임이 된다(고정 사인 아님).
- **조준은 온라인 학습**: 은닉 특징 h에서 "타깃을 중앙에 두는 시점 이동량"으로 가는
  선형 사상을 NLMS로 매 틱 갱신 → 스크립트 에임봇이 아니라 *배우는* 정책.

### 요구사항 ↔ 구현 매핑

| 요구 | 위치 | 방법 |
|------|------|------|
| 처음부터 설계 | `nn/layers.py`, `model/unified.py` | Linear·GRU·tanh/sigmoid/softmax 직접 구현. 외부 모델 0. |
| AI가 움직임(고정 X) | `model/unified.py` `_motion`, `avatar/motion.py` | 모션 헤드 출력을 적분 → 매 프레임 생성, 어댑터가 리그로 매핑. |
| FPS 실시간 | `nn/`(작은 차원), `game/fps_env.py` | 한 틱 수백µs, >1000Hz. 처리량 측정/보고. |
| 조준 빠르게 반응/학습 | `unified.learn_aim` (NLMS) | 은닉 특징→조준 온라인 회귀, 수십~수백 틱에 추적. |
| 모델 1개 | 전체 | 제어·모션·의도가 같은 `h_t` 공유. `brain.agent is streamer.agent`. |
| 말이 빠름(저지연) | `speech/tts.py`, `runtime/streamer.py` | 문장 끝 안 기다리고 글자 청크 즉시 합성, 첫 음성 ~1ms. |
| 종합 게임 / 인게임 튜토리얼만으로 | `game/comprehensive.py`, `cognition/tutorial_learner.py` | 외부 영상·사전지식 없음. 게임이 띄우는 튜토리얼(OCR)에서만 조작 습득 → 배운 만큼만 플레이. |

### 종합 게임을 "인게임 튜토리얼만으로" 학습

특정 게임 한 종류가 아니라, 여러 메커니즘(이동·전투·점프·상호작용·재장전·메뉴)을 가진
**종합 게임**을 다룬다. 에이전트는 조작 체계를 **사전에 모른다**. 외부 튜토리얼 영상도,
별도 학습 단계도 없다 — 오직 게임 자신이 화면에 띄우는 튜토리얼을 OCR로 읽어 습득한다.

```
튜토리얼 학습 전 클리어: 0/6   (조작을 전혀 모름)
[게임의 인게임 튜토리얼 팝업을 화면에서 읽는 중...]
인게임 튜토리얼 학습 후 클리어: 6/6
  ✅ 적 처치 (필요:공격) — 조준 학습 0.15→0.12, 명중 22   ← 전투는 같은 신경망의 온라인 조준
```

`TutorialLearner`가 화면 텍스트에서 의도→키 매핑(`control_scheme()`)을 만들고,
`ComprehensiveGame`은 그 표에 있는 조작만 수행한다. 배우지 않은 조작이 필요한 목표는
실패한다 — "배운 만큼만 플레이".

---

## 제어 루프와 발화의 분리 (FPS 실시간의 핵심)

빠른 제어(매 프레임)와 느린 발화(문장 단위)를 분리한다. 말하는 중에도 게임 반응이
끊기지 않는다.

```python
s = BroadcastStreamer(seed=2025)

# 매 프레임(예: 120~240Hz): 게임 제어 + 아바타 모션
ctrl = s.control_tick(observation)     # FPS 액션 (aim/move/fire ...)
frame = s.render_frame()               # AI가 생성한 아바타 포즈

# 문장 단위(비동기): 발화는 별도로 흘러간다
res = s.on_chat("방금 에임 미쳤다 ㅋㅋ")   # 첫 음성 ~1ms
```

---

## Live2D 아바타 적용 (`avatar/live2d.py`)

보유하신 Live2D Cubism 모델(`.model3.json`/`.moc3`/`.cdi3.json`/...)을 연결하면, AI가
생성한 모션·립싱크·표정이 **그 모델의 실제 파라미터 ID**를 구동한다.

```python
s = BroadcastStreamer()
s.load_avatar("assets/avatar/gothic_lolita/model.model3.json")
frame = s.render_live2d_frame()
# → {"ParamAngleX": 3.7, "ParamEyeLOpen": 1.0, "ParamMouthOpenY": 0.7, "ParamBreath": 0.24, ...}
s.export_live2d_motion("out.jsonl", seconds=5)   # Cubism 런타임이 재생할 프레임
```

- **모델 비종속 매핑**: 눈 깜빡임/립싱크 대상은 `model3.json`의 `Groups`(EyeBlink,
  LipSync)에서 읽는다. 다른 Cubism 모델을 올려도 그 모델의 그룹/파라미터를 따른다.
- 출력 dict를 Cubism 런타임의 `setParameterValueById(id, value)`에 그대로 넣으면 끝.
  실제 메시 렌더링에는 **Live2D Cubism Core/SDK**(Web/Unity/Native)가 필요하다.

> ⚠️ **유료/구매 모델은 git에 커밋하지 않는다.** 모델 바이너리(`.moc3`/텍스처/
> `.physics3.json` 등)는 `.gitignore`로 제외되어 공개 저장소로 새지 않는다. 모델은
> 로컬 `assets/avatar/<name>/` 에 두고 쓰며, 테스트는 자체 제작 중립 픽스처
> (`tests/fixtures/example_cubism/`)를 사용한다. 폴더 사용법은
> `assets/avatar/gothic_lolita/README.md` 참고.

---

## 프로젝트 구조

```
src/broadcast_ai/
  nn/                  # 처음부터 구현한 신경망: layers(Linear/GRU), seed(초기화)
  model/unified.py     # ★ 단일 통합 에이전트 (제어+모션+의도, 온라인 조준학습)
  game/fps_env.py      # FPS 사격장 (실시간 제어 + 조준 학습 시연/측정)
  game/comprehensive.py # 종합 게임 (인게임 튜토리얼만으로 학습해 플레이)
  cognition/           # brain(단일모델 허브), verbalizer(의도→한국어), tutorial_learner
  perception/          # screen(캡처), ocr
  avatar/              # rig, motion(신경망→포즈 어댑터), lipsync, live2d(Cubism 연동)
  speech/tts.py        # 저지연 스트리밍 TTS
  core/                # event_bus, clock(지연 측정)
  gpu/device.py        # GPU 감지·VRAM 예산·상주 적재 계획
  runtime/streamer.py  # 전체 오케스트레이터 + CLI
demo/run_demo.py       # 엔드투엔드 데모
tests/                 # unified/fps/comprehensive/motion/lipsync/tutorial/streamer/live2d (35개)
```

---

## 실제 입출력 백엔드(선택)

신경망은 자체 구현이라 LLM 런타임 의존성이 없다. 음성/화면만 실제 장치로 바꾸려면
(전부 **로컬** 패키지):

```bash
pip install piper-tts            # 저지연 온디바이스 음성
pip install mss pillow pytesseract   # 화면 캡처 + OCR (튜토리얼 학습)
pip install numpy                # nn 백엔드 가속(선택) → 더 높은 Hz
```

---

## 다음 단계 (학습 훅)

현재 가중치는 시드 고정 초기값이라 동작은 재현 가능하지만, 구조는 그대로 학습 가능하다:
- 조준/이동 정책: 온라인 NLMS 외에 보상 기반 RL(자체 구현)로 확장
- 모션 헤드: 사람 모션 통계로 모방학습(`train_motion_imitation` 훅 위치)
- 발화: 의도 헤드 출력을 더 풍부한 텍스트 디코더와 결합

---

## 라이선스

MIT
