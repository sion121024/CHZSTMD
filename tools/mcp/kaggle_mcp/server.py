"""Kaggle MCP 서버.

Claude(또는 다른 MCP 클라이언트)가 Kaggle을 직접 조작할 수 있게 하는 도구 모음.
용도: 제대로 된 대화 말뭉치(뉴스·댓글 제외) 검색/다운로드 → from-scratch 대화 모델을
Kaggle GPU 커널에서 학습 → 상태 확인 → 학습된 체크포인트 회수.

설계
----
* 안정성을 위해 공식 `kaggle` CLI를 subprocess로 호출한다(REST 직접 호출보다 버전
  변화에 강함).
* 인증 키는 코드/채팅에 두지 않는다. 표준 Kaggle 인증을 그대로 쓴다:
    - 파일:   ~/.kaggle/kaggle.json   (chmod 600)
    - 또는 환경변수: KAGGLE_USERNAME, KAGGLE_KEY
  서버는 키 값을 절대 로그/반환하지 않는다.
* stdio 전송으로 동작하므로 Claude Code에 그대로 등록 가능.

실행:  python tools/mcp/kaggle_mcp/server.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("kaggle")

_KAGGLE = shutil.which("kaggle") or "kaggle"
_MAX_OUT = 12000  # 반환 문자열 상한


def _run(args: list[str], timeout: int = 600) -> dict:
    """kaggle CLI 실행. 표준 인증(kaggle.json/환경변수)을 그대로 상속한다."""
    try:
        p = subprocess.run([_KAGGLE, *args], capture_output=True, text=True,
                           timeout=timeout)
    except FileNotFoundError:
        return {"ok": False, "error": "kaggle CLI를 찾을 수 없음. `pip install kaggle`."}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"시간초과({timeout}s): kaggle {' '.join(args)}"}
    out = (p.stdout or "")[:_MAX_OUT]
    err = (p.stderr or "")[:_MAX_OUT]
    return {"ok": p.returncode == 0, "returncode": p.returncode,
            "stdout": out, "stderr": err}


def _creds_present() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


# --------------------------------------------------------------------------- #
@mcp.tool()
def kaggle_auth_status() -> str:
    """Kaggle 인증이 설정됐는지 확인한다(키 값은 노출하지 않음)."""
    if not _creds_present():
        return ("인증 미설정. ~/.kaggle/kaggle.json 을 두거나 "
                "KAGGLE_USERNAME/KAGGLE_KEY 환경변수를 설정하세요. (키는 채팅에 붙여넣지 마세요.)")
    r = _run(["datasets", "list", "-s", "conversation", "-p", "1"])
    user = os.environ.get("KAGGLE_USERNAME") or "(kaggle.json)"
    if r["ok"]:
        return f"인증 정상. 사용자={user}. Kaggle API 호출 성공."
    return f"인증 설정됨(사용자={user})이나 호출 실패:\n{r.get('stderr') or r.get('error')}"


@mcp.tool()
def kaggle_search_datasets(query: str, page: int = 1, page_size: int = 20) -> str:
    """대화 말뭉치 등 데이터셋을 검색한다. 예: query='korean conversation dialogue'.

    뉴스/댓글이 아닌 '대화' 데이터셋을 찾을 때 사용. 결과의 ref(owner/slug)를
    kaggle_download_dataset 에 넘긴다.
    """
    # 이 CLI 버전의 `datasets list`는 페이지 크기 고정(20). page만 지원.
    r = _run(["datasets", "list", "-s", query, "--page", str(page), "--csv"])
    if not r["ok"]:
        return r.get("stderr") or r.get("error") or "검색 실패"
    return r["stdout"] or "결과 없음"


@mcp.tool()
def kaggle_download_dataset(ref: str, dest_dir: str, unzip: bool = True) -> str:
    """데이터셋을 내려받는다. ref='owner/slug', dest_dir=저장 폴더."""
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    args = ["datasets", "download", ref, "-p", dest_dir]
    if unzip:
        args.append("--unzip")
    r = _run(args, timeout=1800)
    if not r["ok"]:
        return r.get("stderr") or r.get("error") or "다운로드 실패"
    files = [f.name for f in Path(dest_dir).iterdir()][:50]
    return f"다운로드 완료 → {dest_dir}\n파일: {files}"


@mcp.tool()
def kaggle_list_my_kernels(page: int = 1, page_size: int = 20) -> str:
    """내 Kaggle 커널(노트북) 목록."""
    r = _run(["kernels", "list", "--mine", "--page", str(page),
              "--page-size", str(page_size), "--csv"])
    return r["stdout"] if r["ok"] else (r.get("stderr") or r.get("error") or "실패")


@mcp.tool()
def kaggle_scaffold_training_kernel(work_dir: str, kernel_slug: str,
                                    title: str, code_path: str,
                                    dataset_sources: list[str] | None = None,
                                    enable_gpu: bool = True,
                                    enable_internet: bool = True) -> str:
    """GPU 학습 커널을 푸시하기 전에 폴더를 구성한다.

    work_dir 안에 kernel-metadata.json 과 학습 스크립트(code_path 복사본)를 만든다.
    kernel_slug 는 'username/my-kernel' 형식(또는 slug만; username 자동 보정 시도).
    dataset_sources 는 학습에 마운트할 데이터셋 ref 목록(['owner/slug', ...]).
    """
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    src = Path(code_path)
    if not src.exists():
        return f"학습 스크립트 없음: {code_path}"
    code_name = src.name
    shutil.copy(src, work / code_name)

    user = os.environ.get("KAGGLE_USERNAME")
    if "/" not in kernel_slug and user:
        kernel_slug = f"{user}/{kernel_slug}"

    meta = {
        "id": kernel_slug,
        "title": title,
        "code_file": code_name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": enable_gpu,
        "enable_internet": enable_internet,
        "dataset_sources": dataset_sources or [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (work / "kernel-metadata.json").write_text(json.dumps(meta, indent=2,
                                                          ensure_ascii=False))
    return (f"스캐폴딩 완료 → {work}\n"
            f"  kernel-metadata.json (id={kernel_slug}, gpu={enable_gpu})\n"
            f"  {code_name}\n다음: kaggle_push_kernel('{work_dir}')")


@mcp.tool()
def kaggle_push_kernel(folder: str) -> str:
    """폴더(kernel-metadata.json + 코드)를 Kaggle 커널로 푸시/실행한다."""
    if not (Path(folder) / "kernel-metadata.json").exists():
        return f"kernel-metadata.json 없음: {folder} (먼저 kaggle_scaffold_training_kernel)"
    r = _run(["kernels", "push", "-p", folder], timeout=600)
    body = (r.get("stdout", "") + "\n" + r.get("stderr", "")).strip()
    return body or ("푸시 성공" if r["ok"] else "푸시 실패")


@mcp.tool()
def kaggle_kernel_status(ref: str) -> str:
    """커널 실행 상태 확인. ref='username/kernel-slug'. (running/complete/error 등)"""
    r = _run(["kernels", "status", ref])
    return (r.get("stdout") or r.get("stderr") or r.get("error") or "상태 미상").strip()


@mcp.tool()
def kaggle_kernel_output(ref: str, dest_dir: str) -> str:
    """완료된 커널의 출력물(학습된 체크포인트 등)을 내려받는다."""
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    r = _run(["kernels", "output", ref, "-p", dest_dir], timeout=1800)
    if not r["ok"]:
        return r.get("stderr") or r.get("error") or "출력물 회수 실패"
    files = [f.name for f in Path(dest_dir).iterdir()][:50]
    return f"출력물 회수 완료 → {dest_dir}\n파일: {files}\n{r.get('stdout','')}"


if __name__ == "__main__":
    mcp.run()
