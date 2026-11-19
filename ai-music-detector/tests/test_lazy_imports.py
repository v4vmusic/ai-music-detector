from __future__ import annotations

import subprocess
import sys

MODEL_LIBRARIES = ("torch", "onnxruntime", "transformers", "sonics", "huggingface_hub")


def test_plain_import_does_not_load_any_model_library() -> None:
    code = (
        "import sys, aidetect; "
        f"loaded = [m for m in sys.modules if m.split('.')[0] in {MODEL_LIBRARIES!r}]; "
        "assert not loaded, loaded; "
        "print(aidetect.__all__[0])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


def test_public_surface_is_explicit() -> None:
    import aidetect

    assert "Detector" in aidetect.__all__
    assert "score_file" not in dir(aidetect)  # methods, not module-level helpers
    assert aidetect.__all__ == sorted(aidetect.__all__)
