"""AI 모션 미리보기 영상 렌더러.

실제 Live2D `.moc3` 메시 렌더는 Cubism Core(런타임)가 필요해 헤드리스 환경에선
그릴 수 없다. 대신 이 도구는 단일 신경망이 *매 프레임 생성한 Live2D 파라미터*
(ParamAngleX, ParamEyeLOpen, ParamMouthOpenY, ...)를 받아 도식형 아바타로 그려
"AI가 실제로 어떻게 움직이는지"를 영상으로 보여준다.

장면:
  1) IDLE     — 호흡/미세 움직임/깜빡임
  2) SPEAKING — 발화에 맞춘 립싱크 + 표정/에너지
  3) FPS AIM  — 움직이는 타깃을 온라인 학습으로 추적(조준 패널)

출력: out/preview.mp4 (+ 실패 시 gif). 캐릭터 텍스처(유료 에셋)는 쓰지 않는다.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402
import imageio  # noqa: E402

from broadcast_ai.runtime import BroadcastStreamer  # noqa: E402
from broadcast_ai.model.unified import Observation  # noqa: E402

W, H = 640, 540
FPS = 30
SKIN = (255, 226, 209)
HAIR = (58, 47, 74)
DRESS = (40, 34, 52)
BG_TOP = (32, 30, 46)
BG_BOT = (18, 17, 28)
ACCENT = (224, 122, 95)

FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
FONT_S = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
FONT_B = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)


def _bg() -> Image.Image:
    img = Image.new("RGB", (W, H), BG_BOT)
    px = img.load()
    for y in range(H):
        t = y / H
        c = tuple(int(BG_TOP[i] * (1 - t) + BG_BOT[i] * t) for i in range(3))
        for x in range(W):
            px[x, y] = c
    return img


def _ellipse(d, cx, cy, rx, ry, **kw):
    d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], **kw)


def draw_avatar(p: dict, cx: float, cy: float, scale: float = 1.0) -> Image.Image:
    """Live2D 파라미터 p로 도식 아바타를 그린 RGBA 레이어를 만든다."""
    L = Image.new("RGBA", (360, 380), (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    ox, oy = 180, 175  # 레이어 내 머리 중심

    breath = p.get("ParamBreath", 0.3)
    body_dx = p.get("ParamBodyAngleX", 0.0) * 1.2
    # 몸통(어깨/드레스) — 호흡으로 살짝 올라옴
    sh_y = oy + 150 - breath * 4
    d.polygon([(ox - 95 + body_dx, 380), (ox - 70 + body_dx, sh_y),
               (ox + 70 + body_dx, sh_y), (ox + 95 + body_dx, 380)], fill=DRESS)
    d.line([(ox + body_dx, sh_y), (ox + body_dx, 380)], fill=(60, 52, 78), width=3)

    # 뒷머리
    _ellipse(d, ox, oy - 6, 122, 132, fill=HAIR)
    # 얼굴
    _ellipse(d, ox, oy, 104, 116, fill=SKIN, outline=(214, 170, 150), width=2)

    # 볼 홍조 — 불투명(반투명으로 칠하면 얼굴 픽셀이 뚫려 어둡게 보임)
    cheek = p.get("ParamCheek", 0.0)
    if cheek > 0.15:
        k = min(1.0, cheek)
        col = tuple(int(SKIN[i] * (1 - 0.5 * k) + (255, 138, 150)[i] * 0.5 * k)
                    for i in range(3)) + (255,)
        for sx in (-1, 1):
            _ellipse(d, ox + sx * 58, oy + 36, 19, 11, fill=col)

    # 눈썹
    brow = p.get("ParamBrowLY", 0.0)
    for sx in (-1, 1):
        bx = ox + sx * 42
        by = oy - 44 - brow * 10
        d.line([(bx - 20, by + sx * 2), (bx + 20, by - sx * 2)], fill=(90, 70, 90), width=5)

    # 눈
    eye_l = p.get("ParamEyeLOpen", 1.0)
    eye_r = p.get("ParamEyeROpen", 1.0)
    smile = p.get("ParamEyeLSmile", 0.0)
    gaze_x = p.get("ParamEyeBallX", 0.0) * 7
    gaze_y = p.get("ParamEyeBallY", 0.0) * 5
    for sx, openness in ((-1, eye_l), (1, eye_r)):
        ex = ox + sx * 42
        ey = oy - 6
        if smile > 0.5:                       # 웃는 눈(^)
            d.arc([ex - 22, ey - 6, ex + 22, ey + 18], 200, 340, fill=(70, 50, 70), width=5)
            continue
        eh = 6 + 34 * max(0.05, openness)
        if openness < 0.12:                   # 감음
            d.line([(ex - 20, ey), (ex + 20, ey)], fill=(70, 50, 70), width=4)
        else:
            _ellipse(d, ex, ey, 22, eh / 2, fill=(255, 255, 255), outline=(120, 100, 120), width=2)
            _ellipse(d, ex + gaze_x, ey + gaze_y, 11, min(11, eh / 2), fill=(74, 60, 96))
            _ellipse(d, ex + gaze_x + 3, ey + gaze_y - 3, 3, 3, fill=(240, 240, 255))

    # 입 — 립싱크
    mouth = p.get("ParamMouthOpenY", 0.0)
    form = p.get("ParamMouthForm", 0.0)
    mx, my = ox, oy + 56
    if mouth < 0.08:
        if form > 0.1:                        # 미소
            d.arc([mx - 24, my - 14, mx + 24, my + 14], 20, 160, fill=(170, 80, 90), width=5)
        else:
            d.line([(mx - 16, my), (mx + 16, my)], fill=(170, 80, 90), width=4)
    else:
        mw = 18 + 14 * max(0.0, form)
        mh = 6 + 34 * mouth
        _ellipse(d, mx, my + mh * 0.2, mw, mh / 2 + 4, fill=(150, 60, 70))
        _ellipse(d, mx, my + mh * 0.45, mw * 0.6, mh * 0.32, fill=(220, 110, 120))  # 혀

    # 앞머리(이마 덮개) — 스웨이에 따라 살짝 흔들림. (옆머리는 뒷머리 실루엣으로 충분)
    hs = p.get("ParamHairSide", 0.0) * 8
    d.pieslice([ox - 118 + hs, oy - 134, ox + 118 + hs, oy + 30], 180, 360, fill=HAIR)

    # 머리 회전(roll = ParamAngleZ)
    angle_z = p.get("ParamAngleZ", 0.0)
    L = L.rotate(-angle_z, resample=Image.BICUBIC, center=(ox, oy))
    if scale != 1.0:
        L = L.resize((int(360 * scale), int(380 * scale)), Image.BICUBIC)
    return L


def compose(streamer, scene, metrics, aim_panel=None):
    img = _bg()
    d = ImageDraw.Draw(img, "RGBA")
    p = streamer.render_live2d_frame(context=_scene_obs(scene))

    # 머리 위치 = ParamAngleX/Y
    ax = p.get("ParamAngleX", 0.0)
    ay = p.get("ParamAngleY", 0.0)
    face_cx = (220 if aim_panel else 320) + ax * 2.2
    layer = draw_avatar(p, face_cx, 250)
    lw, lh = layer.size
    img.paste(layer, (int(face_cx - lw / 2), int(120 - ay * 2.0)), layer)

    # 상단 라벨
    d.rectangle([0, 0, W, 40], fill=(0, 0, 0, 120))
    d.text((16, 9), scene, font=FONT_B, fill=(255, 255, 255))

    # 하단 메트릭(+ 우측 고정 태그)
    d.rectangle([0, H - 34, W, H], fill=(0, 0, 0, 140))
    d.text((16, H - 28), metrics, font=FONT_S, fill=(200, 220, 200))
    tag = "single from-scratch NN -> Live2D"
    tw = d.textlength(tag, font=FONT_S)
    d.text((W - tw - 14, H - 28), tag, font=FONT_S, fill=(150, 150, 175))

    # FPS 조준 패널
    if aim_panel is not None:
        px0, py0, pw = 430, 120, 190
        d.rectangle([px0, py0, px0 + pw, py0 + pw], outline=(90, 200, 120), width=2,
                    fill=(10, 20, 14, 160))
        d.text((px0 + 6, py0 - 22), "FPS aim view", font=FONT_S, fill=(150, 220, 160))
        cx, cy = px0 + pw / 2, py0 + pw / 2
        tx = cx + aim_panel["tx"] * pw / 2 * 0.8
        ty = cy + aim_panel["ty"] * pw / 2 * 0.8
        _ellipse(d, tx, ty, 10, 10, fill=(230, 70, 70))         # 타깃
        chx = cx + aim_panel["cx"] * pw / 2 * 0.8
        chy = cy + aim_panel["cy"] * pw / 2 * 0.8
        d.line([(chx - 12, chy), (chx + 12, chy)], fill=(90, 230, 120), width=2)  # 조준선
        d.line([(chx, chy - 12), (chx, chy + 12)], fill=(90, 230, 120), width=2)
    return img


def _scene_obs(scene: str) -> Observation:
    if scene.startswith("SPEAK"):
        return Observation(chat_activity=1.0, excite_drive=0.8)
    if scene.startswith("FPS"):
        return Observation(target_visible=1.0, focus_drive=0.9)
    return Observation(excite_drive=0.2)


def main() -> None:
    out_dir = ROOT / "out"
    out_dir.mkdir(exist_ok=True)

    s = BroadcastStreamer(seed=2025)
    model_path = ROOT / "assets/avatar/gothic_lolita/model.model3.json"
    if not model_path.exists():
        model_path = ROOT / "tests/fixtures/example_cubism/model.model3.json"
    s.load_avatar(str(model_path))

    frames = []

    # 1) IDLE (2.5s)
    s._speaking = False
    for i in range(int(2.5 * FPS)):
        frames.append(compose(s, "IDLE  (breathing - micro motion - blink)",
                              f"emotion-driven idle | breath/eye/sway all generated by the NN"))

    # 2) SPEAKING (~4s) — 립싱크를 발화 텍스트로 구동
    s._speaking = True
    s._energy = 0.8
    line = "Hey chat! Watch my aim, I'm learning this game live!"
    held = 0
    for ch in line * 2:
        v = s.lipsync.from_text_piece(ch)
        s._viseme_open = v.mouth_open
        s._viseme_wide = v.mouth_wide
        frames.append(compose(s, "SPEAKING  (lip-sync from speech)",
                              "mouth/eyes/brows follow the utterance | TTFA ~1ms"))
        held += 1
        if held >= int(4.2 * FPS):
            break
    s._speaking = False
    s._viseme_open = s._viseme_wide = 0.0

    # 3) FPS AIM (~4s) — 움직이는 타깃을 온라인 학습으로 추적
    cxh = cyh = 0.0
    t = 0.0
    for i in range(int(4.2 * FPS)):
        t += 1.0 / FPS
        tx = 0.6 * math.sin(t * 1.7)
        ty = 0.5 * math.sin(t * 2.3)
        offx, offy = tx - cxh, ty - cyh
        obs = Observation(target_dx=max(-1, min(1, offx)), target_dy=max(-1, min(1, offy)),
                          target_visible=1.0, focus_drive=0.9)
        out = s.agent.tick(obs, dt=1.0 / FPS)
        cxh = max(-1, min(1, cxh + out.control.aim_dx))
        cyh = max(-1, min(1, cyh + out.control.aim_dy))
        err = s.agent.learn_aim(max(-1, min(1, offx)), max(-1, min(1, offy)))
        panel = {"tx": tx, "ty": ty, "cx": cxh, "cy": cyh}
        frames.append(compose(s, "FPS AIM  (online learning, >1000Hz)",
                              f"tracking error: {err:.3f}  (drops as the NN learns to aim)",
                              aim_panel=panel))

    # 인코딩
    mp4 = out_dir / "preview.mp4"
    arrs = [_to_arr(f) for f in frames]
    try:
        imageio.mimsave(mp4, arrs, fps=FPS, codec="libx264", quality=8,
                        macro_block_size=None)
        print(f"saved {mp4} ({len(frames)} frames, {len(frames)/FPS:.1f}s)")
    except Exception as e:
        gif = out_dir / "preview.gif"
        imageio.mimsave(gif, arrs, fps=FPS)
        print(f"mp4 실패({e}) → gif 저장: {gif}")


def _to_arr(img: Image.Image):
    import numpy as np
    return np.asarray(img.convert("RGB"))


if __name__ == "__main__":
    main()
