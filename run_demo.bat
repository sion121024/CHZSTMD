@echo off
REM Windows 실행기 — PYTHONPATH 설정 없이 데모를 바로 실행한다.
REM (demo\run_demo.py 가 내부에서 src 경로를 자동으로 추가한다)
python "%~dp0demo\run_demo.py" %*
