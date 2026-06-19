# Live2D 모델 배치 위치

이 폴더에 **본인이 보유/구매한 Live2D Cubism 모델 파일**을 넣으세요. 유료 에셋이므로
모델 바이너리는 `.gitignore`로 **git에 커밋되지 않습니다**(재배포 방지). 로컬에만 두고
씁니다.

필요한 파일(이름은 `model.model3.json`의 `FileReferences`와 일치하면 됩니다):

```
assets/avatar/gothic_lolita/
  model.model3.json     # 모델 정의 (이 파일이 나머지를 참조)
  model.moc3            # 컴파일된 리그/메시
  model.cdi3.json       # 파라미터/파츠 표시 정보
  model.physics3.json   # 물리 (선택)
  texture/texture_00.png
```

코드 연결:

```python
from broadcast_ai.runtime import BroadcastStreamer
s = BroadcastStreamer()
s.load_avatar("assets/avatar/gothic_lolita/model.model3.json")
frame = s.render_live2d_frame()          # {"ParamAngleX": 12.3, "ParamEyeLOpen": 1.0, ...}
s.export_live2d_motion("out.jsonl", seconds=5)  # Cubism 런타임이 재생할 프레임
```

`render_live2d_frame()`가 돌려주는 dict를 Cubism 런타임의
`setParameterValueById(id, value)`에 그대로 넣으면, AI가 생성한 모션·립싱크·표정이
이 모델에 적용됩니다. 실제 메시 렌더링에는 Live2D Cubism Core/SDK가 필요합니다
(NOTICE.md 참고).
