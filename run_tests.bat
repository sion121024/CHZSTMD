@echo off
REM Windows 테스트 실행기 — tests\conftest.py 가 src 경로를 자동 추가하므로
REM PYTHONPATH 없이 그대로 돈다.
python -m pip install pytest -q
python -m pytest %*
