# 아바타 모델 고지 (Avatar Model Notice)

## ゴスロリちゃん (Gothic Lolita-chan) — Live2D Cubism 모델

- 형식: **Live2D Cubism** 모델 (`.moc3` / `.model3.json` / `.cdi3.json` /
  `.physics3.json` / 텍스처)
- 파라미터: 표준 Cubism 30종 (ParamAngleX/Y/Z, ParamEyeLOpen/ROpen,
  ParamMouthOpenY, ParamMouthForm, ParamBrow*, ParamBreath, ParamHair* 등)
- 그룹: `EyeBlink` = [ParamEyeLOpen, ParamEyeROpen], `LipSync` = [ParamMouthOpenY]

이 모델은 **프로젝트 소유자가 보유/구매한 Live2D 에셋**입니다. 본 저장소는 이 모델을
"AI가 생성한 모션/립싱크로 구동"하기 위한 어댑터(`src/broadcast_ai/avatar/live2d.py`)
연동 목적으로만 사용합니다. 모델의 저작권은 원저작자에게 있으며, 재배포·상업적
이용 등은 **구매 시 동의한 원저작자/판매처의 라이선스 약관**을 따릅니다.

> 참고: 업로드된 파일의 내부 폴더명은 `ゴスロリちゃん無料版`(무료판)으로 표기되어
> 있습니다. 이는 해당 파일의 폴더 이름일 뿐입니다. 정식(유료) 버전 파일로
> 교체하려면 그 파일을 같은 경로에 넣고 `model.model3.json` 참조만 맞추면 됩니다.

### 렌더링/배포 시
- 실제 `.moc3` 메시를 화면에 그리려면 **Live2D Cubism Core / SDK**(Web · Unity ·
  Native 중 택1)가 필요합니다. Cubism Core는 Live2D사 라이선스를 따르며 본 저장소에
  포함되지 않습니다 — 뷰어 사용 시 직접 받아 넣어야 합니다.
- 유료 에셋이므로 공개 저장소에 모델 바이너리를 올리기 전, 라이선스상 재배포가
  허용되는지 반드시 확인하세요. (본 저장소에서는 `.gitignore`로 모델 바이너리 커밋을
  막을 수 있습니다 — 아래 README 참고.)

우리 코드가 하는 일: AI가 만든 `Pose` → 이 모델의 **실제 파라미터 ID 값**으로 매핑한
프레임(`{ "ParamAngleX": 12.3, "ParamEyeLOpen": 1.0, ... }`)을 생성합니다. 이 프레임을
Cubism 런타임의 `setParameterValueById(id, value)`에 그대로 넣으면 됩니다.
