"""Core transcription functions — usable as a library without GUI dependencies."""

from __future__ import annotations

import functools
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf


DEFAULT_SAMPLE_RATE = 16_000


def list_audio_devices() -> None:
    """Print available audio input devices."""
    devices = sd.query_devices()
    for index, device in enumerate(devices):
        if device.get("max_input_channels", 0) > 0:
            print(
                f"{index}: {device['name']} (inputs: {device['max_input_channels']}, "
                f"default SR: {device['default_samplerate']})"
            )


def record_audio(
    duration: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = 1,
    device: Optional[int] = None,
) -> np.ndarray:
    """Record audio for a fixed duration. Returns numpy array of samples."""
    if duration <= 0:
        raise ValueError("Duration must be a positive number of seconds.")
    try:
        sd.check_input_settings(device=device, channels=channels, samplerate=sample_rate)
    except Exception as exc:
        raise RuntimeError(f"Failed to access input device: {exc}") from exc

    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        device=device,
    )
    sd.wait()
    return recording


def save_wav(audio: np.ndarray, path: Path, sample_rate: int = DEFAULT_SAMPLE_RATE) -> Path:
    """Write audio samples to a WAV file, collapsing to mono if necessary."""
    if audio.ndim > 1 and audio.shape[1] > 1:
        audio = np.mean(audio, axis=1, keepdims=True)
    sf.write(file=str(path), data=audio, samplerate=sample_rate)
    return path


@functools.lru_cache(maxsize=None)
def load_whisper_model(model_name: str, device: Optional[str] = None) -> Any:
    """Load a Whisper model (cached)."""
    import whisper
    return whisper.load_model(model_name, device=device)


def transcribe_audio(
    audio_path: Path,
    model_name: str = "base",
    language: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Transcribe a WAV file using Whisper.

    Returns:
        {"text": "...", "segments": [...], "language": "en"}
    """
    import whisper  # noqa: F811

    model = load_whisper_model(model_name, device)
    task = "translate" if language not in {None, "en"} else "transcribe"
    result = model.transcribe(str(audio_path), language=language, task=task)

    return {
        "text": result.get("text", "").strip(),
        "segments": result.get("segments", []),
        "language": result.get("language", language or "unknown"),
    }
