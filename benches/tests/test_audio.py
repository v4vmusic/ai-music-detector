from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from benches._shared.audio import decode_window
from benches._shared.constants import WARNING_SHORT_TRACK_PADDED
from benches._shared.errors import AudioDecodeError

WriteWav = Callable[[Path, float], Path]


def test_decode_window_math(tmp_path: Path, write_wav: WriteWav) -> None:
    audio = write_wav(tmp_path / "ten.wav", 10.0)
    window = decode_window(audio, window_seconds=3.0, offset_seconds=2.0)
    assert window.samples.shape == (48_000,)
    assert window.samples.dtype == np.float32
    assert window.sample_rate == 16_000
    assert window.offset_seconds == 2.0
    assert window.window_seconds == 3.0
    assert np.all(np.abs(window.samples) <= 1.0)


def test_decode_window_offset_matches_full_decode(tmp_path: Path, write_wav: WriteWav) -> None:
    audio = write_wav(tmp_path / "ten.wav", 10.0)
    full = decode_window(audio, window_seconds=10.0)
    window = decode_window(audio, window_seconds=3.0, offset_seconds=2.0)
    np.testing.assert_array_equal(window.samples, full.samples[32_000:80_000])


def test_decode_window_pads_short_track_with_warning(tmp_path: Path, write_wav: WriteWav) -> None:
    audio = write_wav(tmp_path / "six.wav", 6.0)
    with pytest.warns(UserWarning, match=WARNING_SHORT_TRACK_PADDED):
        window = decode_window(audio, window_seconds=10.0)
    assert window.samples.shape == (160_000,)
    np.testing.assert_array_equal(window.samples[96_000:], np.zeros(64_000, dtype=np.float32))
    assert WARNING_SHORT_TRACK_PADDED in window.warnings


def test_decode_window_rejects_under_five_seconds(tmp_path: Path, write_wav: WriteWav) -> None:
    audio = write_wav(tmp_path / "short.wav", 4.9)
    with pytest.raises(AudioDecodeError, match="5"):
        decode_window(audio, window_seconds=3.0)


def test_non_audio_input_raises_audio_decode_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"this is not audio\n")
    with pytest.raises(AudioDecodeError, match="ffmpeg"):
        decode_window(bad, window_seconds=3.0)


def test_decode_window_rejects_negative_offset(tmp_path: Path, write_wav: WriteWav) -> None:
    audio = write_wav(tmp_path / "ten.wav", 10.0)
    with pytest.raises(ValueError, match="offset"):
        decode_window(audio, window_seconds=3.0, offset_seconds=-1.0)


def test_decode_window_rejects_non_positive_window(tmp_path: Path, write_wav: WriteWav) -> None:
    audio = write_wav(tmp_path / "ten.wav", 10.0)
    with pytest.raises(ValueError, match="window"):
        decode_window(audio, window_seconds=0.0)
