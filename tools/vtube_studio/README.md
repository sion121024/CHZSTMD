# VTube Studio 플러그인 — AI가 youling 조종

우리 단일 신경망이 만든 모션·립싱크·감정을 **VTube Studio 안의 youling**에 실시간
주입합니다. VTS가 렌더·물리·표정을 담당하고, "어떻게 움직일지"는 우리 AI가 정합니다.

## 준비
1. **VTube Studio** 실행 → 설정(톱니) → **"Start API"**(플러그인 허용) 켜기. 포트 `8001`.
2. 의존성:
   ```bat
   pip install -r tools\vtube_studio\requirements.txt
   ```
3. 실행:
   ```bat
   python tools\vtube_studio\vts_driver.py
   ```
4. 처음 실행하면 VTS에 **플러그인 허용 팝업** → **Allow** 클릭.
   (토큰이 `tools\vtube_studio\vts_token.txt`에 저장되어 다음부턴 자동)

→ VTS의 youling이 우리 AI대로 숨쉬고·깜빡이고·말하고·감정 표정이 바뀝니다.

## 어떻게 동작하나
- AI가 만든 머리각도/입/눈/눈썹 → VTS **기본 트래킹 파라미터**(FaceAngleX, MouthOpen,
  EyeOpenLeft/Right, MouthSmile, Brows …)에 주입.
- youling의 `youling.vtube.json`이 이 파라미터들을 모델에 매핑하므로 그대로 움직입니다.
- 감정 표정: VTS의 **핫키(표정)** 이름에 happy/smile/angry/surprise 등이 있으면 감정에
  맞춰 자동 트리거. (핫키가 없으면 파라미터로만 표현)

## 안 될 때
- **연결 실패**: VTS에서 API가 켜져 있는지(포트 8001), 방화벽 확인.
- **인증 실패**: `vts_token.txt` 삭제 후 다시 실행 → 허용 팝업 다시.
- **표정 안 바뀜**: VTS에서 표정을 '핫키'로 등록하고 이름을 happy/angry 등으로.
  콘솔에 출력된 "VTS 핫키" 목록을 보고 이름을 맞추세요.

## A안(웹 뷰어) vs 이 플러그인(VTS)
- 웹 뷰어: 브라우저에서 우리가 직접 렌더. 추가 앱 불필요.
- VTS 플러그인: VTS의 고품질 렌더/물리/표정 + 우리 AI 조종. (이 폴더)
둘은 **같은 AI**를 씁니다. 송출은 둘 다 OBS로 가능.
