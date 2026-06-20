@echo off
REM youling 웹 뷰어 실행기 (Windows). 추가 설치 불필요.
REM 브라우저가 잠깐 빈 화면이면 1~2초 뒤 새로고침하세요(모델 로딩).
start "" http://localhost:8765/tools/web_viewer/index.html
python "%~dp0tools\web_viewer\server.py"
pause
