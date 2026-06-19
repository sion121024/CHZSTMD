"""립싱크 입모양 검증."""

from broadcast_ai.avatar.lipsync import LipSync


def test_open_vowel_opens_mouth_more_than_closed():
    ls = LipSync(smoothing=1.0)  # 즉시 반영
    a = ls.from_text_piece("아")
    ls.reset()
    i = ls.from_text_piece("이")
    assert a.mouth_open > i.mouth_open      # '아'가 더 크게 벌어짐
    assert i.mouth_wide > a.mouth_wide      # '이'가 더 넓게


def test_audio_amplitude_drives_open():
    ls = LipSync(smoothing=1.0)
    loud = ls.from_audio_amplitude(0.9)
    ls.reset()
    quiet = ls.from_audio_amplitude(0.1)
    assert loud.mouth_open > quiet.mouth_open


def test_rest_closes_mouth():
    ls = LipSync(smoothing=1.0)
    ls.from_text_piece("아")
    rest = ls.rest()
    assert rest.mouth_open == 0.0


def test_smoothing_prevents_jumps():
    ls = LipSync(smoothing=0.4)
    v1 = ls.from_text_piece("아")
    # 한 스텝에 목표(1.0)까지 가지 않고 부드럽게 접근.
    assert 0.0 < v1.mouth_open < 1.0
