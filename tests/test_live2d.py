"""Live2D 연동 검증 — 자체 제작 중립 픽스처 사용(유료 에셋 불필요).

실제 구매 모델은 .gitignore로 제외되며 로컬 assets/ 경로에 둔다. 같은 표준 Cubism
파라미터 인터페이스이므로 어댑터 동작은 동일하다.
"""

from pathlib import Path

from broadcast_ai.avatar.live2d import Live2DModel, Live2DAvatar
from broadcast_ai.avatar.rig import Pose
from broadcast_ai.runtime import BroadcastStreamer

FIXTURE = Path(__file__).parent / "fixtures" / "example_cubism" / "model.model3.json"


def test_loads_parameters_and_groups():
    m = Live2DModel.load(FIXTURE)
    assert len(m.parameter_ids) >= 18
    assert m.eyeblink_ids == ["ParamEyeLOpen", "ParamEyeROpen"]
    assert m.lipsync_ids == ["ParamMouthOpenY"]
    assert m.has("ParamAngleX")


def test_pose_maps_to_real_param_ids_in_range():
    m = Live2DModel.load(FIXTURE)
    av = Live2DAvatar(m)
    frame = av.apply(Pose(head_yaw=1.0, head_pitch=-1.0, eye_blink=1.0,
                          mouth_open=0.8, smile=0.5, breath=0.5))
    # 출력 키는 모두 모델에 존재하는 실제 파라미터 ID여야 한다.
    assert all(m.has(k) for k in frame)
    # 머리 각도는 ±30도 범위로 클램프.
    assert -30 <= frame["ParamAngleX"] <= 30
    # 눈 감음(blink=1) → EyeOpen 0.
    assert frame["ParamEyeLOpen"] == 0.0
    assert frame["ParamEyeROpen"] == 0.0
    # 립싱크는 모델의 LipSync 그룹(ParamMouthOpenY)을 구동.
    assert frame["ParamMouthOpenY"] == 0.8


def test_lipsync_override_uses_group_id():
    m = Live2DModel.load(FIXTURE)
    av = Live2DAvatar(m)
    frame = av.apply(Pose(), lipsync_open=0.95)
    assert frame["ParamMouthOpenY"] == 0.95


def test_unknown_params_are_omitted():
    """모델에 없는 파라미터는 출력하지 않는다(다른 모델 호환)."""
    m = Live2DModel.load(FIXTURE)
    av = Live2DAvatar(m)
    frame = av.apply(Pose(arm_l=0.5, arm_r=-0.5))
    # 이 픽스처에는 팔 파라미터가 없음 → 팔 관련 키 없음, 몸각도로만 반영.
    assert "ParamArmL" not in frame
    assert "ParamBodyAngleZ" in frame


def test_streamer_drives_live2d_and_exports(tmp_path):
    s = BroadcastStreamer(seed=5)
    model = s.load_avatar(str(FIXTURE))
    assert model.parameter_ids
    # AI가 매 프레임 실제 파라미터를 생성.
    f1 = s.render_live2d_frame()
    f2 = s.render_live2d_frame()
    assert "ParamAngleX" in f1
    assert f1 != f2                       # 매 프레임 다름(AI 생성)
    # JSONL 내보내기 — Cubism 런타임이 재생할 프레임.
    out = tmp_path / "motion.jsonl"
    n = s.export_live2d_motion(str(out), seconds=0.5, fps=30)
    assert n == 15
    assert out.read_text().count("ParamAngleX") == 15
