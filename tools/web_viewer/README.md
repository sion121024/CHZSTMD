# youling 웹 뷰어 (A안) — 우리 AI가 구동

브라우저에서 youling이 **우리 단일 신경망이 생성한** 호흡·깜빡임·표정·립싱크대로
움직입니다. 추가 설치 불필요(표준 라이브러리만). 인터넷은 처음 한 번 CDN(PixiJS·
Cubism Core) 로드에만 필요합니다.

## 실행 (Windows)

저장소 폴더에서 **`run_viewer.bat` 더블클릭**. 또는:

```bat
python tools\web_viewer\server.py
```
그다음 브라우저에서: **http://localhost:8765/tools/web_viewer/index.html**

- 화면이 잠깐 비면 1~2초 뒤 **새로고침**(모델 로딩 시간).
- 종료: 콘솔 창에서 **Ctrl+C**.

## 구조

```
브라우저(index.html)  ──30fps── GET /frame ──►  server.py(우리 AI)
   pixi-live2d-display                            BroadcastStreamer
   youling.moc3 렌더                              render_live2d_frame()
       ▲ setParameterValueById(id, value) ◄──── { "ParamAngleX": .., "ParamMouthOpenY": .. }
```

- `server.py`: 한 포트에서 정적 파일(뷰어+모델)과 `/frame`(AI 파라미터 JSON)을 서빙.
- `index.html`: 모델을 그리고 매 프레임 AI 파라미터를 주입.

## 모델 파일명이 다르면

기본은 `youling.model3.json`. 다른 이름이면 URL 뒤에 붙이세요:
```
http://localhost:8765/tools/web_viewer/index.html?model=내모델.model3.json
```
모델 폴더는 `assets/avatar/gothic_lolita/` 입니다(여기에 youling 파일들이 있어야 함).

## 표정 바꾸기 (.exp3.json)

뷰어에서 **숫자키 `1`~`9`, `0`** 으로 표정을 전환합니다(youling의 expression1~10).
`Esc` 또는 `` ` `` 로 해제. 화면 좌상단에 현재 표정이 표시됩니다.

## 무료버전 표식(워터마크) 제거

서버가 모델의 파트/파라미터 이름에서 표식 키워드(무료·체험·sample·watermark·ロゴ 등)를
찾아 **자동으로 숨깁니다**(해당 파트 불투명도 0).

자동으로 안 잡히면(이름이 다르면) 직접 찾습니다:
```bat
python inspect_model.py "C:\경로\youling폴더"
```
→ 파트/파라미터 목록이 나옵니다. 표식으로 의심되는 줄을 알려주시면 정확히 끄도록
연결해 드립니다.

## 방송 송출

OBS → **브라우저 소스**에 위 URL을 넣으면 투명 배경으로 youling이 송출됩니다.

## 안 될 때

- **빈 화면**: 1~2초 뒤 새로고침. 그래도면 F12(개발자도구) → Console 의 빨간 에러를 알려주세요.
- **모델 로드 실패**: `?model=정확한파일명.model3.json` 확인. 폴더에 `.model3.json`/`.moc3`/텍스처(`*.4096/`)가 다 있는지 확인.
- **CDN 차단 환경**: PixiJS/Cubism Core를 로컬에 받아 `index.html`의 `<script src>`를 로컬 경로로 바꾸면 오프라인에서도 됩니다.
