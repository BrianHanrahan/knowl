"""VoiceRecorder — background-threaded mic recording for voice-first input."""

from __future__ import annotations

import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd

from knowl.voice.transcribe import DEFAULT_SAMPLE_RATE, save_wav


class VoiceRecorder:
    """Manage a start/stop microphone recording session.

    Uses a background thread via sounddevice's callback API so the
    calling thread (e.g. UI) stays responsive.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = 1,
        device: Optional[int] = None,
        silence_threshold: float = 0.01,
        silence_duration: float = 2.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self._stream: Optional[sd.InputStream] = None
        self._frames: list[np.ndarray] = []
        self._lock = threading.Lock()

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        """Begin recording from the microphone."""
        if self._stream is not None:
            raise RuntimeError("Recording already in progress.")

        try:
            sd.check_input_settings(
                device=self.device,
                channels=self.channels,
                samplerate=self.sample_rate,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to access input device: {exc}") from exc

        self._frames = []

        def callback(indata: np.ndarray, frames: int, time: object, status: object) -> None:
            if status:
                print(f"Input stream status: {status}", file=sys.stderr)
            with self._lock:
                self._frames.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            callback=callback,
            device=self.device,
        )
        self._stream.start()

    def stop(self) -> Path:
        """Stop recording and return the path to a WAV file with the captured audio."""
        if self._stream is None:
            # Return an empty WAV
            tmp = tempfile.NamedTemporaryFile(prefix="knowl_voice_", suffix=".wav", delete=False)
            path = Path(tmp.name)
            tmp.close()
            empty = np.empty((0, self.channels), dtype="float32")
            save_wav(empty, path, self.sample_rate)
            return path

        self._stream.stop()
        self._stream.close()
        self._stream = None

        with self._lock:
            if not self._frames:
                audio = np.empty((0, self.channels), dtype="float32")
            else:
                audio = np.concatenate(self._frames, axis=0)
            self._frames = []

        tmp = tempfile.NamedTemporaryFile(prefix="knowl_voice_", suffix=".wav", delete=False)
        path = Path(tmp.name)
        tmp.close()
        save_wav(audio, path, self.sample_rate)
        return path
