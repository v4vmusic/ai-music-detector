from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from benches._shared.audio import decode_window
from benches._shared.constants import WARNING_LOOP_FILLED, WARNING_SHORT_TRACK_PADDED
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


def test_decode_window_peak_normalizes_out_of_range_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 6 s at 16 kHz, with peak 1.5 (> 1.0) -> should scale down to fit [-1, 1].
    samples = np.linspace(-0.5, 0.5, 96_000, dtype=np.float32)
    samples[100] = 1.5
    samples[200] = -1.2
    peak = float(np.max(np.abs(samples)))

    def fake_decode_pcm(path: Path) -> NDArray[np.float32]:
        return samples.copy()

    monkeypatch.setattr("benches._shared.audio._decode_pcm", fake_decode_pcm)
    audio = tmp_path / "hot.wav"
    audio.write_bytes(b"placeholder")

    with pytest.warns(UserWarning, match="peak_normalized_to_fit_range"):
        window = decode_window(audio, window_seconds=6.0)

    assert window.samples.shape == (96_000,)
    assert "peak_normalized_to_fit_range" in window.warnings
    assert float(np.max(np.abs(window.samples))) == pytest.approx(1.0)
    np.testing.assert_allclose(window.samples, samples / peak, rtol=0, atol=1e-6)


def test_decode_window_loop_fills_only_the_short_remainder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = np.linspace(-0.5, 0.5, 96_000, dtype=np.float32)

    def fake_decode_pcm(path: Path) -> NDArray[np.float32]:
        return samples.copy()

    monkeypatch.setattr("benches._shared.audio._decode_pcm", fake_decode_pcm)
    audio = tmp_path / "six.wav"
    audio.write_bytes(b"placeholder")

    with pytest.warns(UserWarning, match=WARNING_LOOP_FILLED):
        window = decode_window(audio, window_seconds=10.0, fill_mode="loop-fill")

    assert window.samples.shape == (160_000,)
    assert window.warnings == (WARNING_LOOP_FILLED,)
    np.testing.assert_array_equal(window.samples[:96_000], samples)
    np.testing.assert_array_equal(window.samples[96_000:], samples[:64_000])


def test_decode_window_loop_fill_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = np.linspace(-0.4, 0.4, 96_000, dtype=np.float32)
    monkeypatch.setattr("benches._shared.audio._decode_pcm", lambda path: samples.copy())
    audio = tmp_path / "three.wav"
    audio.write_bytes(b"placeholder")

    with pytest.warns(UserWarning, match=WARNING_LOOP_FILLED):
        first = decode_window(audio, window_seconds=10.0, fill_mode="loop-fill")
    with pytest.warns(UserWarning, match=WARNING_LOOP_FILLED):
        second = decode_window(audio, window_seconds=10.0, fill_mode="loop-fill")

    np.testing.assert_array_equal(first.samples, second.samples)


def test_decode_window_loop_fill_leaves_full_segments_unchanged(
    tmp_path: Path, write_wav: WriteWav
) -> None:
    audio = write_wav(tmp_path / "ten.wav", 10.0)
    plain = decode_window(audio, window_seconds=3.0, offset_seconds=2.0)
    looped = decode_window(audio, window_seconds=3.0, offset_seconds=2.0, fill_mode="loop-fill")
    np.testing.assert_array_equal(looped.samples, plain.samples)
    assert looped.warnings == ()
    assert plain.warnings == ()


def test_decode_window_explicit_zero_pad_matches_baseline(
    tmp_path: Path, write_wav: WriteWav
) -> None:
    audio = write_wav(tmp_path / "six.wav", 6.0)
    with pytest.warns(UserWarning, match=WARNING_SHORT_TRACK_PADDED):
        baseline = decode_window(audio, window_seconds=10.0)
    with pytest.warns(UserWarning, match=WARNING_SHORT_TRACK_PADDED):
        explicit = decode_window(audio, window_seconds=10.0, fill_mode="zero-pad")
    np.testing.assert_array_equal(explicit.samples, baseline.samples)
    assert explicit.warnings == (WARNING_SHORT_TRACK_PADDED,)


def test_decode_window_loop_fill_rejects_an_empty_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = np.zeros(96_000, dtype=np.float32)
    monkeypatch.setattr("benches._shared.audio._decode_pcm", lambda path: samples.copy())
    audio = tmp_path / "six.wav"
    audio.write_bytes(b"placeholder")

    with pytest.raises(AudioDecodeError, match="no audio"):
        decode_window(audio, window_seconds=3.0, offset_seconds=10.0, fill_mode="loop-fill")


def test_decode_window_placement_resolves_offset_from_duration(
    tmp_path: Path, write_wav: WriteWav
) -> None:
    audio = write_wav(tmp_path / "ten.wav", 10.0)
    middle = decode_window(audio, window_seconds=4.0, placement="middle")
    fixed = decode_window(audio, window_seconds=4.0, offset_seconds=3.0)
    assert middle.offset_seconds == pytest.approx(3.0)
    np.testing.assert_array_equal(middle.samples, fixed.samples)


def test_decode_window_placement_end_avoids_padding_for_a_full_window(
    tmp_path: Path, write_wav: WriteWav
) -> None:
    audio = write_wav(tmp_path / "six.wav", 6.0)
    window = decode_window(audio, window_seconds=6.0, placement="end")
    assert window.offset_seconds == 0.0
    assert window.warnings == ()
    assert window.samples.shape == (96_000,)


def test_decode_window_explicit_fixed_placement_uses_the_offset(
    tmp_path: Path, write_wav: WriteWav
) -> None:
    audio = write_wav(tmp_path / "ten.wav", 10.0)
    placed = decode_window(audio, window_seconds=4.0, offset_seconds=3.0, placement="fixed")
    expected = decode_window(audio, window_seconds=4.0, offset_seconds=3.0)
    np.testing.assert_array_equal(placed.samples, expected.samples)
    assert placed.offset_seconds == 3.0


def test_decode_window_placement_ignores_a_fixed_offset(
    tmp_path: Path, write_wav: WriteWav
) -> None:
    audio = write_wav(tmp_path / "ten.wav", 10.0)
    placed = decode_window(audio, window_seconds=4.0, offset_seconds=99.0, placement="end")
    expected = decode_window(audio, window_seconds=4.0, offset_seconds=6.0)
    assert placed.offset_seconds == pytest.approx(6.0)
    np.testing.assert_array_equal(placed.samples, expected.samples)
