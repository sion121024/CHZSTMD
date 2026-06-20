"""Live2D Cubism 모델 연동 — AI 모션을 실제 아바타 파라미터로 매핑.

이 모듈은 사용자가 올린 Live2D 모델(.model3.json/.moc3/.cdi3.json/...)을 읽어,
우리 단일 신경망이 만든 `Pose`를 그 모델의 *실제 파라미터 ID 값*으로 변환한다.
출력 프레임(`{paramId: value}`)을 Cubism 런타임의 `setParameterValueById`에 그대로
넣으면 아바타가 AI가 생성한 대로 움직인다.

모델에 의존하지 않게 설계: 눈 깜빡임/립싱크 대상은 model3.json의 `Groups`(EyeBlink,
LipSync)에서 읽어온다. 다른 Cubism 모델을 올려도 그 모델의 그룹/파라미터를 따른다.
.moc3 메시 렌더링 자체는 Cubism Core(런타임)가 담당하며 여기서는 다루지 않는다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .rig import Pose


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


@dataclass
class Live2DModel:
    """파싱된 Live2D 모델 메타데이터."""

    root: Path
    moc_path: Path
    texture_paths: list[Path]
    physics_path: Path | None
    parameter_ids: list[str] = field(default_factory=list)
    parameter_names: dict[str, str] = field(default_factory=dict)
    eyeblink_ids: list[str] = field(default_factory=list)
    lipsync_ids: list[str] = field(default_factory=list)
    part_ids: list[str] = field(default_factory=list)
    part_names: dict[str, str] = field(default_factory=dict)
    # 표정 파일: [(이름, Path), ...]
    expressions: list = field(default_factory=list)

    # 무료/체험 표식(워터마크) 추정 키워드 (KO/EN/JP)
    WATERMARK_KW = [
        "무료", "체험", "샘플", "표식", "워터마크", "데모",
        "free", "trial", "sample", "watermark", "logo", "demo",
        "ロゴ", "透かし", "体験", "サンプル", "無料", "ウォーターマーク",
    ]

    def has(self, param_id: str) -> bool:
        return param_id in self.parameter_ids

    def watermark_candidates(self) -> list[tuple[str, str, str]]:
        """이름/ID에 표식 키워드가 들어간 파트·파라미터 후보를 찾는다.

        반환: [(kind('part'|'param'), id, name), ...]
        """
        kw = [k.lower() for k in self.WATERMARK_KW]
        out: list[tuple[str, str, str]] = []
        for pid in self.part_ids:
            name = self.part_names.get(pid, pid)
            if any(k in (pid + " " + name).lower() for k in kw):
                out.append(("part", pid, name))
        for pid in self.parameter_ids:
            name = self.parameter_names.get(pid, pid)
            if any(k in (pid + " " + name).lower() for k in kw):
                out.append(("param", pid, name))
        return out

    # cdi3(DisplayInfo)가 없거나 비어 있을 때 가정하는 표준 Cubism 파라미터.
    DEFAULT_PARAMS = [
        "ParamAngleX", "ParamAngleY", "ParamAngleZ", "ParamEyeBallX", "ParamEyeBallY",
        "ParamBodyAngleX", "ParamBodyAngleZ", "ParamBreath", "ParamBrowLY", "ParamBrowRY",
        "ParamEyeLOpen", "ParamEyeROpen", "ParamEyeLSmile", "ParamEyeRSmile", "ParamCheek",
        "ParamMouthOpenY", "ParamMouthForm", "ParamHairFront", "ParamHairSide", "ParamHairBack",
    ]

    @staticmethod
    def _find_model3(path: str | Path) -> Path | None:
        """파일이면 그대로, 폴더(또는 잘못된 파일경로)면 안에서 *.model3.json 탐색."""
        p = Path(path)
        if p.is_file() and p.name.endswith(".model3.json"):
            return p
        search_dir = p if p.is_dir() else p.parent
        if not search_dir.exists():
            return None
        cands = sorted(search_dir.rglob("*.model3.json"))
        return cands[0] if cands else None

    @staticmethod
    def diagnose(path: str | Path) -> str:
        """무엇을 찾았는지 사람이 읽을 수 있게 보고(인식 실패 디버깅용)."""
        p = Path(path)
        d = p if p.is_dir() else p.parent
        if not d.exists():
            return f"경로가 존재하지 않음: {d}"
        files = sorted(f.name for f in d.iterdir()) if d.is_dir() else []
        m3 = [str(x) for x in d.rglob("*.model3.json")]
        return f"폴더={d}\n  파일들={files[:40]}\n  발견된 .model3.json={m3 or '없음 ❌'}"

    @classmethod
    def load(cls, model3_path: str | Path) -> "Live2DModel":
        found = cls._find_model3(model3_path)
        if not found:
            raise FileNotFoundError(
                f"Live2D 모델(.model3.json)을 인식하지 못했습니다.\n"
                f"입력: {model3_path}\n{cls.diagnose(model3_path)}\n"
                f"→ 해결: 모델 *폴더* 경로를 주세요(파일명이 달라도 자동으로 찾습니다). "
                f"폴더 안에 .model3.json 이 실제로 있는지 확인하세요.")
        p = found
        root = p.parent
        spec = json.loads(p.read_text(encoding="utf-8"))
        refs = spec.get("FileReferences", {})
        moc = root / refs.get("Moc", "")
        textures = [root / t for t in refs.get("Textures", [])]
        physics = root / refs["Physics"] if refs.get("Physics") else None

        eyeblink: list[str] = []
        lipsync: list[str] = []
        for g in spec.get("Groups", []):
            if g.get("Name") == "EyeBlink":
                eyeblink = list(g.get("Ids", []))
            elif g.get("Name") == "LipSync":
                lipsync = list(g.get("Ids", []))

        param_ids: list[str] = []
        param_names: dict[str, str] = {}
        part_ids: list[str] = []
        part_names: dict[str, str] = {}
        disp = refs.get("DisplayInfo")
        if disp and (root / disp).exists():
            cdi = json.loads((root / disp).read_text(encoding="utf-8"))
            for prm in cdi.get("Parameters", []):
                param_ids.append(prm["Id"])
                param_names[prm["Id"]] = prm.get("Name", prm["Id"])
            for prt in cdi.get("Parts", []):
                part_ids.append(prt["Id"])
                part_names[prt["Id"]] = prt.get("Name", prt["Id"])
        if not param_ids:
            # cdi3 없음/비었음 → 표준 파라미터로 가정해 어댑터가 동작하게 한다.
            param_ids = list(cls.DEFAULT_PARAMS)

        # 표정: model3.json 참조 우선, 없으면 폴더의 *.exp3.json 보강.
        expressions: list = []
        for ex in refs.get("Expressions", []):
            f = root / ex.get("File", "")
            expressions.append((ex.get("Name", f.stem), f))
        if not expressions:
            for f in sorted(root.glob("*.exp3.json")):
                expressions.append((f.stem, f))

        return cls(
            root=root, moc_path=moc, texture_paths=textures, physics_path=physics,
            parameter_ids=param_ids, parameter_names=param_names,
            eyeblink_ids=eyeblink, lipsync_ids=lipsync,
            part_ids=part_ids, part_names=part_names, expressions=expressions,
        )

    def summary(self) -> str:
        return (
            f"Live2D[{self.moc_path.name}] 파라미터 {len(self.parameter_ids)}개 · "
            f"EyeBlink={self.eyeblink_ids} · LipSync={self.lipsync_ids} · "
            f"텍스처 {len(self.texture_paths)}장"
        )


# 표준 Cubism 파라미터 ↔ 우리 Pose 채널 매핑.
# (값, 범위)는 Cubism 관례. 모델에 그 파라미터가 있을 때만 출력한다.
class Live2DAvatar:
    """Pose → Live2D 파라미터 프레임 변환기."""

    def __init__(self, model: Live2DModel) -> None:
        self.model = model

    def apply(self, pose: Pose, lipsync_open: float | None = None,
              lipsync_form: float | None = None) -> dict[str, float]:
        m = self.model
        out: dict[str, float] = {}

        def put(pid: str, value: float, lo: float, hi: float) -> None:
            if m.has(pid):
                out[pid] = round(_clamp(value, lo, hi), 4)

        # 머리 각도 (deg)
        put("ParamAngleX", pose.head_yaw * 60.0, -30, 30)
        put("ParamAngleY", pose.head_pitch * 60.0, -30, 30)
        put("ParamAngleZ", pose.head_roll * 60.0, -30, 30)
        # 시선 (눈동자)
        put("ParamEyeBallX", pose.head_yaw * 2.0, -1, 1)
        put("ParamEyeBallY", pose.head_pitch * 2.0, -1, 1)
        # 몸 회전 / 기울임
        put("ParamBodyAngleX", pose.body_sway * 80.0, -10, 10)
        put("ParamBodyAngleZ", (pose.arm_r - pose.arm_l) * 20.0 + pose.head_roll * 10.0, -10, 10)
        # 호흡
        put("ParamBreath", pose.breath, 0, 1)
        # 눈썹
        put("ParamBrowLY", pose.brow_raise, -1, 1)
        put("ParamBrowRY", pose.brow_raise, -1, 1)
        # 미소(눈) / 볼 홍조
        put("ParamEyeLSmile", pose.smile, 0, 1)
        put("ParamEyeRSmile", pose.smile, 0, 1)
        put("ParamCheek", pose.smile * 0.7, 0, 1)
        # 머리카락 흔들림(물리가 없거나 보강용) — 몸/머리 움직임에 따라 살짝
        put("ParamHairFront", -pose.head_yaw * 1.5, -1, 1)
        put("ParamHairSide", -pose.body_sway * 3.0, -1, 1)
        put("ParamHairBack", pose.body_sway * 2.0, -1, 1)

        # 눈 깜빡임 — 모델의 EyeBlink 그룹을 따른다(1=뜸, 0=감음).
        eye_open = 1.0 - _clamp(pose.eye_blink, 0.0, 1.0)
        for pid in (m.eyeblink_ids or ["ParamEyeLOpen", "ParamEyeROpen"]):
            put(pid, eye_open, 0, 1)

        # 립싱크 — 모델의 LipSync 그룹을 따른다.
        mouth = pose.mouth_open if lipsync_open is None else lipsync_open
        for pid in (m.lipsync_ids or ["ParamMouthOpenY"]):
            put(pid, mouth, 0, 1)
        form = pose.mouth_wide if lipsync_form is None else lipsync_form
        put("ParamMouthForm", form * 0.5 + pose.smile * 0.5, -1, 1)

        return out
