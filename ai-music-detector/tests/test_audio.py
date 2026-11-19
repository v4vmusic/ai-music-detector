from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from aidetect.audio import decode_window, decode_window_bytes, is_audio_payload
from aidetect.constants import (
    WARNING_LOOP_FILLED,
    WARNING_PEAK_NORMALIZED,
    WARNING_SHORT_TRACK_PADDED,
)
from aidetect.errors import AudioDecodeError

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


def test_decode_window_positioning_is_duration_relative(
    tmp_path: Path, write_wav: WriteWav
) -> None:
    audio = write_wav(tmp_path / "ten.wav", 10.0)
    middle = decode_window(audio, window_seconds=4.0, placement="middle")
    fixed = decode_window(audio, window_seconds=4.0, offset_seconds=3.0)
    assert middle.offset_seconds == pytest.approx(3.0)
    np.testing.assert_array_equal(middle.samples, fixed.samples)


def test_decode_window_peak_normalizes_out_of_range_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = np.linspace(-0.5, 0.5, 96_000, dtype=np.float32)
    samples[100] = 1.5
    peak = float(np.max(np.abs(samples)))
    monkeypatch.setattr("aidetect.audio._decode_pcm", lambda path: samples.copy())
    audio = tmp_path / "hot.wav"
    audio.write_bytes(b"placeholder")

    with pytest.warns(UserWarning, match="peak_normalized_to_fit_range"):
        window = decode_window(audio, window_seconds=6.0)

    assert WARNING_PEAK_NORMALIZED in window.warnings
    assert float(np.max(np.abs(window.samples))) == pytest.approx(1.0)
    np.testing.assert_allclose(window.samples, samples / peak, rtol=0, atol=1e-6)


def test_decode_window_loop_fills_short_remainder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = np.linspace(-0.5, 0.5, 96_000, dtype=np.float32)
    monkeypatch.setattr("aidetect.audio._decode_pcm", lambda path: samples.copy())
    audio = tmp_path / "six.wav"
    audio.write_bytes(b"placeholder")

    with pytest.warns(UserWarning, match=WARNING_LOOP_FILLED):
        window = decode_window(audio, window_seconds=10.0, fill_mode="loop-fill")

    assert window.warnings == (WARNING_LOOP_FILLED,)
    np.testing.assert_array_equal(window.samples[:96_000], samples)
    np.testing.assert_array_equal(window.samples[96_000:], samples[:64_000])


def test_decode_window_bytes_matches_file(tmp_path: Path, write_wav: WriteWav) -> None:
    audio = write_wav(tmp_path / "ten.wav", 10.0)
    from_file = decode_window(audio, window_seconds=10.0, placement="middle")
    from_bytes = decode_window_bytes(audio.read_bytes(), window_seconds=10.0, placement="middle")
    np.testing.assert_array_equal(from_file.samples, from_bytes.samples)
    assert from_bytes.offset_seconds == pytest.approx(from_file.offset_seconds)


def test_decode_window_bytes_rejects_empty_payload() -> None:
    with pytest.raises(ValueError, match="empty"):
        decode_window_bytes(b"", window_seconds=3.0)


def test_decode_window_rejects_invalid_window_and_offset(
    tmp_path: Path, write_wav: WriteWav
) -> None:
    audio = write_wav(tmp_path / "ten.wav", 10.0)
    with pytest.raises(ValueError, match="window"):
        decode_window(audio, window_seconds=0.0)
    with pytest.raises(ValueError, match="offset"):
        decode_window(audio, window_seconds=3.0, offset_seconds=-1.0)


def test_decode_pcm_rejects_incomplete_float_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    class _Result:
        returncode = 0
        stdout = b"123"
        stderr = b""

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _Result())
    audio = tmp_path / "odd.wav"
    audio.write_bytes(b"placeholder")
    with pytest.raises(AudioDecodeError, match="incomplete"):
        decode_window(audio, window_seconds=3.0)


def test_is_audio_payload_accepts_common_containers(tmp_path: Path, write_wav: WriteWav) -> None:
    assert is_audio_payload(write_wav(tmp_path / "track.wav", 0.2).read_bytes())
    assert is_audio_payload(b"ID3\x04\x00\x00\x00\x00\x00\x00payload")
    assert is_audio_payload(b"\xff\xfb\x90\x64payload")
    assert is_audio_payload(b"OggS\x00\x02payload")
    assert is_audio_payload(b"fLaCpayload")
    assert is_audio_payload(b"\x00\x00\x00\x18ftypM4A payload")
    assert is_audio_payload(b"\xff\xf1\x50\x80payload")
    assert is_audio_payload(b"\x1a\x45\xdf\xa3payload")
    assert is_audio_payload(b"FORM\x00\x00\x00\x0cAIFFpayload")


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"this is not audio\n",
        b"<html><body>login</body></html>",
        b'{"error": "not found"}',
        b"RIFF\x00\x00\x00\x00JUNKnot-wave",
    ],
)
def test_is_audio_payload_rejects_non_audio(payload: bytes) -> None:
    assert not is_audio_payload(payload)
