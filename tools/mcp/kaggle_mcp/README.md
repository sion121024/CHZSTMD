# Kaggle MCP 서버

Claude(MCP 클라이언트)가 **Kaggle을 직접 조작**하게 해 주는 도구 모음. 목적:
제대로 된 대화 데이터로 **from-scratch 한국어 대화 모델을 Kaggle GPU에서 학습** →
체크포인트를 받아 아바타에 연결.

## 무엇을 하나

8개 MCP 도구:

| 도구 | 설명 |
|------|------|
| `kaggle_auth_status` | 인증 설정 확인(키 값 노출 안 함) |
| `kaggle_search_datasets` | 데이터셋 검색 |
| `kaggle_download_dataset` | 데이터셋 다운로드 |
| `kaggle_list_my_kernels` | 내 커널 목록 |
| `kaggle_scaffold_training_kernel` | 학습 커널 폴더 구성(메타데이터+스크립트) |
| `kaggle_push_kernel` | 커널 푸시/실행(GPU) |
| `kaggle_kernel_status` | 실행 상태 |
| `kaggle_kernel_output` | 출력물(체크포인트) 회수 |

## 설치

```bash
pip install -r tools/mcp/kaggle_mcp/requirements.txt   # mcp, kaggle
```

## 인증 (키는 채팅·저장소에 두지 마세요)

표준 Kaggle 인증을 그대로 씁니다. **API 키를 코드/채팅/깃에 넣지 마세요.**

- **Windows**: `C:\Users\<사용자>\.kaggle\kaggle.json`
- **Linux/macOS**: `~/.kaggle/kaggle.json` (그리고 `chmod 600 ~/.kaggle/kaggle.json`)

`kaggle.json` 내용은 Kaggle → *Account → API → Create New Token* 에서 받습니다:
```json
{"username":"<사용자명>","key":"<API_KEY>"}
```
또는 환경변수 `KAGGLE_USERNAME`, `KAGGLE_KEY` 로 줄 수도 있습니다.

> 🔒 **키가 노출됐다면 즉시 폐기/재발급** (Account → API → *Expire Token* 후 새 토큰).
> `.gitignore`가 `kaggle.json`을 커밋에서 막습니다.

## Claude Code에 등록

이 저장소 루트의 `.mcp.json`이 서버를 자동 등록합니다(키는 로컬 `kaggle.json`에서 읽음).
수동 등록도 가능:
```bash
claude mcp add kaggle -- python tools/mcp/kaggle_mcp/server.py
```

## 엔드투엔드 흐름 (유창한 대화 모델 만들기)

데이터는 **HuggingFace의 제대로 된 멀티턴 대화셋**을 커널 안에서 받습니다
(뉴스·댓글 아님; 기본값 `junelee/sharegpt_deepl_ko`, 교체 가능). Kaggle엔 양질의
한국어 텍스트 대화셋이 드물어 이렇게 합니다. 학습은 Kaggle GPU에서 돕니다.

1. `kaggle_auth_status()` — 인증 확인
2. `kaggle_scaffold_training_kernel(work_dir="kernel_build", kernel_slug="haru-dialogue", title="Haru dialogue", code_path="tools/mcp/kaggle_mcp/payload/train_dialogue.py", enable_gpu=True, enable_internet=True)`
   - 커널이 HF에서 대화셋을 받아 from-scratch GPT를 학습(기본 ~30M, 6000스텝).
3. `kaggle_push_kernel("kernel_build")` — Kaggle GPU에서 실행 시작
4. `kaggle_kernel_status("<사용자>/haru-dialogue")` — `running`→`complete`
5. `kaggle_kernel_output("<사용자>/haru-dialogue", "checkpoint")` — `ckpt.pt`, `spm.model` 회수
6. 아바타에 연결:
   ```python
   from broadcast_ai.runtime import BroadcastStreamer
   s = BroadcastStreamer()
   s.load_dialogue_model("checkpoint")   # 성공 시 학습 모델로 응답, 실패 시 템플릿 폴백
   print(s.on_chat("하루야 오늘 뭐해?").text)
   ```

## 데이터셋 바꾸기 (뉴스·댓글 제외, 대화만)

`payload/train_dialogue.py`의 `--hf_dataset` 기본값을 바꾸거나 스캐폴딩 시 인자로
넘기면 됩니다. 멀티턴 대화/문답 스키마(`conversations`, `messages`, Q/A 등)를
자동 파싱합니다.

## 유창함에 대한 정직한 메모

무료 Kaggle GPU로 from-scratch 학습은 **짧고 자연스러운 대화** 수준까지는 가지만,
완전한 자유 대화는 데이터·스텝·모델 크기를 키워야 합니다(`--steps`, `--dim`,
`--layers`, `--vocab`). 사람 목소리는 학습/복제하지 않습니다(목소리는 별도 기성 TTS).
