# 방송용 AI 버추얼 스트리머 (CHZSTMD)

**진짜 사람처럼 움직이고, 말이 빠르고 유창하며, GPU 친화적이고, 게임을 누가 안
알려줘도 화면 튜토리얼만 보고 스스로 배우는** 방송용 AI 버추얼 캐릭터.

> 설계 제약 (요구사항 그대로)
> - **API 없음** — 외부 API를 호출하지 않는다. 모든 추론은 **완전 로컬(온디바이스)**.
> - **반응속도** — 토큰 스트리밍 → 점진적 음성 합성 → 립싱크/모션을 **동시에** 굴려
>   첫 음성(TTFA)을 수십 ms 안에 낸다.
> - **단일 모델** — 대화·게임이해·튜토리얼학습을 **하나의 로컬 모델**이 멀티태스크로 처리.

이 저장소는 외부 의존성 0(표준 라이브러리만)으로 **지금 바로 실행/테스트**된다.
실제 로컬 모델·TTS·OCR 가중치를 꽂으면 그대로 프로덕션 파이프라인이 된다(폴백 → 실백엔드).

---

## 빠른 시작

```bash
# 의존성 설치 불필요 — 코어는 표준 라이브러리만으로 동작
python3 demo/run_demo.py          # 엔드투엔드 데모
python3 -m pytest                 # 테스트 (pip install pytest 후)
python3 -m broadcast_ai.runtime.streamer   # 대화형 CLI (PYTHONPATH=src)
```

데모 출력 예 (발췌):

```
시청자> 하루야 안녕!
하루> ㅋㅋ 하루야 안녕! 그치, 나도 완전 공감해. 자 계속 가보자고!
   첫 음성 13ms · 발화 2292ms · 청크 4개 · ✅ 지연 예산 내

📚 지금까지 배운 조작: 이동(WASD), 점프(SPACE), 공격(LMB), 방어(RMB), 상호작용(E), 재장전(R)
```

---

## 아키텍처

한 번의 반응이 흐르는 경로 — 각 단계가 다음 단계를 **블로킹하지 않는다**:

```
입력(채팅 / 게임상황 / 화면 텍스트)
   │
   ▼  perception/  ── 화면 캡처(mss) → OCR(pytesseract) ── 게임 화면을 "본다"
   ▼  cognition/Brain ── 단일 LocalModel을 task 태그로 호출 (대화/게임/튜토리얼)
   │        └ TutorialLearner ── 화면 안내문을 조작 지식으로 누적
   ▼  (토큰 스트리밍) ───────────────────────────────────────────────┐
   ▼  speech/StreamingTTS ── 글자 청크가 차면 즉시 합성  ← 첫 음성 지연(TTFA) 측정
   ▼  avatar/LipSync ── 텍스트/오디오 → 입모양(viseme)
   ▼  avatar/MotionSynth ── 호흡·깜빡임·미세흔들림·제스처 (순수 CPU 수학, 60fps)
   ▼  avatar/Rig ── Pose → 렌더러(VRM/Live2D/Unity)로 송출
```

### 요구사항 ↔ 구현 매핑

| 요구 | 구현 위치 | 방법 |
|------|-----------|------|
| 진짜 사람처럼 움직임 | `avatar/motion.py` | 호흡 사인파 + 지수분포 깜빡임 + 의사-펄린 미세흔들림 + 말하기 제스처. 정지 상태도 "살아있게". |
| 말이 빠름 (저지연) | `speech/tts.py`, `runtime/streamer.py` | 문장 끝을 안 기다리고 글자 청크가 차면 즉시 합성. 첫 토큰부터 입을 연다. `core/clock.py`로 TTFA 측정. |
| GPU 친화 | `gpu/device.py`, `config.py` | 모델 1회 적재 후 **상주**(reload 없음), int4 양자화, VRAM 예산 기반 배치 자동조정, KV 캐시 재사용. CUDA/Metal 없으면 CPU 폴백. |
| 말이 유창 | `cognition/persona.py`, `model/local_model.py` | 안정적 페르소나 프리픽스(KV 캐시 대상) + 구어체 시스템 프롬프트 + 스트리밍 디코딩. |
| 튜토리얼 자동 학습 | `perception/`, `cognition/tutorial_learner.py` | 화면 OCR 텍스트 → 조작 의도/키 추출 → 누적 지식 → 같은 단일 모델 컨텍스트로 주입. |

---

## "단일 모델"의 의미

`LocalModel` 인스턴스는 **하나뿐**이고, `Brain`이 작업에 따라 `task` 태그(`chat` /
`game` / `tutorial`)만 바꿔 같은 모델에 흘려보낸다. 작업별로 모델을 따로 두지 않으므로

- 메모리에 모델 1개만 상주 → VRAM 절약, GPU 친화
- 모델 전환(reload)이 없어 → 매 턴 반응이 빠름
- 페르소나/시스템 프리픽스의 KV 캐시를 모든 작업이 공유

```python
assert streamer.brain.model is streamer.model   # 항상 같은 인스턴스
```

---

## 실제 백엔드 꽂기

코어는 폴백으로 동작하지만, 아래를 설치/지정하면 자동으로 실백엔드로 전환된다
(전부 **로컬** 패키지, 외부 API 아님):

```bash
# 단일 로컬 LLM (GGUF, CPU/GPU 친화)
pip install llama-cpp-python
export BAI_MODEL_PATH=/path/to/model.gguf

# 저지연 온디바이스 TTS
pip install piper-tts

# 화면 인식 / OCR (게임 튜토리얼 자동 학습)
pip install mss pillow pytesseract
```

주요 환경변수(`config.py`): `BAI_MODEL_PATH`, `BAI_CTX`, `BAI_MAX_REPLY`,
`BAI_GPU`, `BAI_QUANT`, `BAI_VRAM_GB`, `BAI_TTS_CHUNK`, `BAI_FPS`, `BAI_TTFA`.

---

## 프로젝트 구조

```
src/broadcast_ai/
  config.py            # 저지연/로컬 우선 기본값 설정
  gpu/device.py        # GPU 감지·VRAM 예산·양자화·상주 적재 계획
  core/                # event_bus(비동기 파이프라인), clock(지연 측정)
  model/local_model.py # 단일 로컬 모델 (스트리밍 + 오프라인 폴백)
  cognition/           # persona, brain(단일모델 허브), tutorial_learner
  perception/          # screen(캡처), ocr
  avatar/              # rig, motion(사람같은 모션), lipsync
  runtime/streamer.py  # 전체 오케스트레이터 + CLI
demo/run_demo.py       # 엔드투엔드 데모
tests/                 # 모션/립싱크/튜토리얼/통합 테스트 (17개)
```

---

## 테스트

```bash
pip install pytest && python3 -m pytest -q
# 17 passed
```

모션이 절대 완전정지하지 않음, 입모양이 모음에 반응, 튜토리얼 중복 제거,
첫 음성이 전체 텍스트 완성 전에 나옴(저지연) 등을 검증한다.

---

## 라이선스

MIT
