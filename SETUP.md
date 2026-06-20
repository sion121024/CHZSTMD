# 새 컴퓨터에서 시작하기 (Setup)

제대로 된 PC(특히 NVIDIA GPU)에서 돌리는 가이드. Windows 기준이며, 리눅스/맥도
`python3`/경로만 바꾸면 동일합니다.

---

## 0. 준비물
- **Python 3.10+** (https://www.python.org — 설치 시 "Add to PATH" 체크)
- **git**
- (선택) **NVIDIA GPU** — 대화 모델 로컬 학습·빠른 추론에 사용

확인:
```bat
python --version
git --version
```

## 1. 내려받기
```bat
git clone -b claude/broadcast-ai-virtual-character-rd8bvv <저장소URL> CHZSTMD
cd CHZSTMD
```
(이미 받았다면: `git pull`)

## 2. 바로 실행 — 데모 (설치 0)
```bat
python demo\run_demo.py
```
→ 단일 모델·FPS 실시간·AI 모션·종합게임 자가학습이 콘솔에서 한 번에 돕니다.

## 3. youling 띄우기 — 웹 뷰어 (AI가 조종)
1. **youling 모델 폴더**의 파일들을 `assets\avatar\gothic_lolita\` 에 넣기
   (`youling.model3.json`, `youling.moc3`, `youling.cdi3.json`, `*.4096`, 표정들 등)
2. 실행:
```bat
run_viewer.bat
```
   또는 `python tools\web_viewer\server.py`
3. 브라우저: **http://localhost:8765/tools/web_viewer/index.html**

youling이 **숨쉬고·깜빡이고·말하고(립싱크)·감정 따라 표정**이 바뀝니다.
- 숫자키 `1`~`0`: 표정 수동 전환 / `Esc`: 해제
- 무료 표식은 자동 숨김 (안 잡히면 `python inspect_model.py "youling폴더"`로 찾기)
- OBS "브라우저 소스"에 위 URL → 방송 송출

## 4. (GPU 있으면) 대화 모델 직접 학습 — Kaggle 없이 로컬에서
```bat
pip install torch sentencepiece
python tools\mcp\kaggle_mcp\payload\train_dialogue.py --out checkpoint --steps 6000
```
- 인터넷 되면 HuggingFace 대화셋을 받아 학습(뉴스/댓글 아님). 안 되면 내장 시드로 폴백.
- 학습 후 아바타에 연결:
```bat
python -c "from broadcast_ai.runtime import BroadcastStreamer; s=BroadcastStreamer(); print(s.load_dialogue_model('checkpoint')); print(s.on_chat('하루야 안녕').text)"
```
> 큰 GPU면 `--dim 512 --layers 8 --steps 20000` 등으로 키워 더 유창하게.

## 5. (선택) 진짜 목소리 — 로컬 TTS
```bat
pip install piper-tts
```
사람 목소리를 새로 학습하지 않고 기성 한국어 음성으로 발화합니다.

## 6. (선택) Kaggle로 학습 — MCP
GPU가 없으면 Kaggle GPU 사용. `tools/mcp/kaggle_mcp/README.md` 참고.
> 🔒 Kaggle 키는 `~/.kaggle/kaggle.json`(또는 `%USERPROFILE%\.kaggle\kaggle.json`)에만.
> 채팅·저장소에 넣지 마세요. 이전에 노출된 키는 **폐기/재발급**하세요.

---

## 좋은 PC가 주는 것
| 항목 | 노트북 | 좋은 PC(GPU) |
|------|--------|--------------|
| 코어 데모/뷰어 | OK (1,000Hz+) | 더 부드러움 |
| 대화 모델 학습 | Kaggle 필요 | **로컬에서 직접·빠르게** |
| 더 큰 모델/유창함 | 제한 | **가능** |
| 추론 속도 | CPU | GPU 가속 |

## 자주 쓰는 명령
```bat
python demo\run_demo.py                     REM 전체 데모
run_viewer.bat                              REM youling 웹 뷰어
python try_avatar.py "youling폴더"          REM 모델 인식 테스트
python inspect_model.py "youling폴더"       REM 파라미터/파트/표식 보기
python -m pytest                            REM 테스트(35개)
```

막히면 그때 뜨는 메시지(또는 브라우저 F12 콘솔의 빨간 에러)를 그대로 보여주세요.
