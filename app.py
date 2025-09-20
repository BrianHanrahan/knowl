import argparse
import functools
import json
import os
import subprocess
import sys
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
from dotenv import load_dotenv


DEFAULT_SAMPLE_RATE = 16_000
TRANSCRIPT_DIR = Path.home() / "Documents" / "transcriptions"
DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-3.1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_CLEAN_PROMPT = (
    "You are an expert copy editor tasked with cleaning a noisy automatic speech transcript. "
    "Improve grammar, punctuation, and clarity while keeping the speakers’ intent. If you insert or replace "
    "words that were unclear, wrap your replacements in square brackets, e.g. ‘[the store]’. Preserve existing "
    "timestamps and speaker labels exactly, but only when the speaker first speaks or when the speaker changes. "
    "Return only the revised transcript text."
)

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")


def list_audio_devices() -> None:
    """Print available audio input devices."""
    devices = sd.query_devices()
    for index, device in enumerate(devices):
        if device.get("max_input_channels", 0) > 0:
            print(
                f"{index}: {device['name']} (inputs: {device['max_input_channels']}, "
                f"default SR: {device['default_samplerate']})"
            )


def record_audio(duration: float, sample_rate: int, channels: int, device: Optional[int]) -> np.ndarray:
    """Record audio for a fixed duration (CLI usage)."""
    if duration <= 0:
        raise ValueError("Duration must be a positive number of seconds.")

    try:
        sd.check_input_settings(device=device, channels=channels, samplerate=sample_rate)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to access input device: {exc}") from exc

    print(f"Recording for {duration} seconds at {sample_rate} Hz...")
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        device=device,
    )
    sd.wait()
    print("Recording complete.")
    return recording


def write_wav(audio: np.ndarray, sample_rate: int, path: Path) -> None:
    """Persist audio samples to WAV, collapsing to mono if necessary."""
    if audio.ndim > 1 and audio.shape[1] > 1:
        audio = np.mean(audio, axis=1, keepdims=True)
    sf.write(file=str(path), data=audio, samplerate=sample_rate)


@functools.lru_cache(maxsize=None)
def load_whisper_model(model_name: str, device: Optional[str]):
    print(f"Loading Whisper model '{model_name}'...")
    return whisper.load_model(model_name, device=device)


@functools.lru_cache(maxsize=1)
def _load_diarization_pipeline(token: str):
    from pyannote.audio import Pipeline  # Imported lazily to avoid dependency unless needed.

    return Pipeline.from_pretrained(DIARIZATION_MODEL_ID, use_auth_token=token)


def load_diarization_pipeline(token: str):
    if not token:
        raise RuntimeError(
            "Speaker diarization requires a Hugging Face access token. Provide one via --diarization-token "
            "or the PYANNOTE_AUTH_TOKEN environment variable."
        )
    return _load_diarization_pipeline(token)


def diarize_audio(audio_path: Path, token: Optional[str]) -> List[Dict[str, float]]:
    token = token or os.getenv("PYANNOTE_AUTH_TOKEN")
    pipeline = load_diarization_pipeline(token)
    diarization = pipeline(str(audio_path))
    segments: List[Dict[str, float]] = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "speaker": speaker,
            }
        )
    segments.sort(key=lambda item: (item["start"], item["end"]))
    if not segments:
        raise RuntimeError("Diarization did not produce any speaker segments.")
    return segments


def assign_speakers(whisper_segments: List[Dict], diarization_segments: List[Dict]) -> List[Dict]:
    assigned: List[Dict] = []
    for seg in whisper_segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        text = seg.get("text", "").strip()
        best_speaker = None
        best_overlap = 0.0
        for diar in diarization_segments:
            overlap = min(end, diar["end"]) - max(start, diar["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = diar["speaker"]
        assigned.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "speaker": best_speaker or "Speaker",
            }
        )
    return assigned


def format_timestamp(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_transcript_segments(segments: List[Dict], speaker_map: Optional[Dict[str, str]] = None) -> str:
    if not segments:
        return ""
    speaker_map = speaker_map or {}
    lines: List[str] = []
    for seg in segments:
        speaker = seg.get("speaker") or "Speaker"
        speaker_name = speaker_map.get(speaker, speaker)
        timestamp = format_timestamp(seg.get("start", 0.0))
        text = seg.get("text", "").strip()
        if text:
            lines.append(f"[{timestamp}] {speaker_name}: {text}")
    return "\n".join(lines).strip()


def parse_speaker_labels(values: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid speaker label '{item}'. Use SPEAKER_00=Alice format.")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Invalid speaker label '{item}'. Use SPEAKER_00=Alice format.")
        mapping[key] = value
    return mapping


def transcribe_pipeline(
    audio_path: Path,
    model_name: str,
    language: Optional[str],
    device: Optional[str],
    diarize: bool,
    diarization_token: Optional[str],
) -> Dict[str, object]:
    model = load_whisper_model(model_name, device)
    result = model.transcribe(str(audio_path), language=language)

    raw_text = result.get("text", "").strip()
    whisper_segments = result.get("segments", [])

    segments: List[Dict] = []
    diarized = False
    if diarize:
        diarization_segments = diarize_audio(audio_path, diarization_token)
        segments = assign_speakers(whisper_segments, diarization_segments)
        diarized = bool(segments)

    return {
        "raw_text": raw_text,
        "segments": segments,
        "diarized": diarized,
    }


def save_transcription_text(text: str) -> Path:
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    path = TRANSCRIPT_DIR / f"{timestamp}.txt"
    counter = 1
    while path.exists():
        path = TRANSCRIPT_DIR / f"{timestamp}_{counter}.txt"
        counter += 1
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def _openai_cleanup_direct(transcript: str, prompt: str, model: str) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:  # noqa: F401
        raise RuntimeError(
            "The 'openai' package is not installed. Install it with 'pip install openai'."
        ) from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; cannot clean transcript with OpenAI.")

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You polish speech transcripts produced by automatic recognition systems, consult web search when"
                    " needed to resolve unclear references, and return only the cleaned transcript with any additions"
                    " in square brackets."
                ),
            },
            {
                "role": "user",
                "content": f"{prompt.strip()}\n\nTranscript:\n{transcript.strip()}",
            },
        ],
        temperature=0.2,
        tools=[{"type": "web_search"}],
    )

    cleaned = (response.output_text or "").strip()
    if not cleaned:
        raise RuntimeError("OpenAI did not return any content.")
    return cleaned


def run_openai_cleanup_worker(input_path: Path, output_path: Path) -> None:
    try:
        payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
        transcript = payload["transcript"]
        prompt = payload["prompt"]
        model = payload["model"]
        cleaned = _openai_cleanup_direct(transcript, prompt, model)
        output = {"cleaned_text": cleaned}
    except Exception as exc:  # noqa: BLE001
        output = {"error": str(exc)}

    Path(output_path).write_text(json.dumps(output), encoding="utf-8")


def clean_transcript_with_openai(transcript: str, prompt: str, model: str) -> str:
    with tempfile.TemporaryDirectory(prefix="openai_cleanup_") as temp_dir:
        input_path = Path(temp_dir) / "input.json"
        output_path = Path(temp_dir) / "output.json"
        payload = {"transcript": transcript, "prompt": prompt, "model": model}
        input_path.write_text(json.dumps(payload), encoding="utf-8")

        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--openai-clean-worker",
                "--openai-clean-input",
                str(input_path),
                "--openai-clean-output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            stderr = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(
                f"OpenAI cleanup subprocess failed (exit code {result.returncode}): {stderr or 'unknown error'}"
            )

        if not output_path.exists():
            raise RuntimeError("OpenAI cleanup subprocess did not produce an output file.")

        data = json.loads(output_path.read_text(encoding="utf-8"))
        if data.get("error"):
            raise RuntimeError(data["error"])
        cleaned = (data.get("cleaned_text") or "").strip()
        if not cleaned:
            raise RuntimeError("OpenAI cleanup returned no content.")
        return cleaned


class AudioRecorder:
    """Manage a start/stop microphone recording session."""

    def __init__(self, sample_rate: int, channels: int, device: Optional[int]):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self._stream: Optional[sd.InputStream] = None
        self._frames: List[np.ndarray] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._stream is not None:
            raise RuntimeError("Recording already in progress.")

        try:
            sd.check_input_settings(device=self.device, channels=self.channels, samplerate=self.sample_rate)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to access input device: {exc}") from exc

        self._frames = []

        def callback(indata, frames, time, status):  # noqa: ARG001
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

    def stop(self) -> np.ndarray:
        if self._stream is None:
            return np.empty((0, self.channels), dtype="float32")

        self._stream.stop()
        self._stream.close()
        self._stream = None

        with self._lock:
            if not self._frames:
                return np.empty((0, self.channels), dtype="float32")
            audio = np.concatenate(self._frames, axis=0)
            self._frames = []
        return audio


def launch_ui(args: argparse.Namespace) -> None:
    try:
        from PySide6 import QtCore, QtWidgets
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError("PySide6 is required for the UI. Install it with 'pip install PySide6'.") from exc

    class TranscriptionWorker(QtCore.QObject):
        finished = QtCore.Signal(object, object)  # result, error

        def __init__(
            self,
            audio: np.ndarray,
            sample_rate: int,
            args: argparse.Namespace,
            diarize: bool,
            diarization_token: Optional[str],
            speaker_map: Optional[Dict[str, str]] = None,
        ) -> None:
            super().__init__()
            self.audio = audio
            self.sample_rate = sample_rate
            self.args = args
            self.diarize = diarize
            self.diarization_token = diarization_token
            self.speaker_map = speaker_map or {}

        @QtCore.Slot()
        def run(self) -> None:
            result: Dict[str, object] = {"raw_text": "", "segments": [], "diarized": False}
            error: Optional[str] = None
            temp_path: Optional[Path] = None

            try:
                if self.audio.size == 0:
                    raise RuntimeError("No audio captured. Try recording again.")

                with tempfile.NamedTemporaryFile(prefix="whisper_recording_", suffix=".wav", delete=False) as tmp:
                    temp_path = Path(tmp.name)

                write_wav(self.audio, self.sample_rate, temp_path)
                result = transcribe_pipeline(
                    temp_path,
                    self.args.model,
                    self.args.language,
                    self.args.whisper_device,
                    self.diarize,
                    self.diarization_token,
                )

                if result.get("diarized"):
                    formatted = format_transcript_segments(result["segments"], self.speaker_map)
                else:
                    formatted = result.get("raw_text", "")

                saved_path = save_transcription_text(formatted)
                result["formatted_text"] = formatted
                result["saved_path"] = str(saved_path)
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                result["formatted_text"] = ""
                result["saved_path"] = ""
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)

            self.finished.emit(result, error)

    class OpenAICleanupWorker(QtCore.QObject):
        finished = QtCore.Signal(object, object)  # cleaned_text, error

        def __init__(self, transcript: str, prompt: str, model: str) -> None:
            super().__init__()
            self.transcript = transcript
            self.prompt = prompt
            self.model = model

        @QtCore.Slot()
        def run(self) -> None:
            try:
                cleaned_text = clean_transcript_with_openai(self.transcript, self.prompt, self.model)
                self.finished.emit(cleaned_text, None)
            except Exception as exc:  # noqa: BLE001
                self.finished.emit("", str(exc))

    class PromptDialog(QtWidgets.QDialog):
        def __init__(self, parent: Optional[QtWidgets.QWidget], default_prompt: str) -> None:
            super().__init__(parent)
            self.setWindowTitle("OpenAI Cleanup Prompt")
            layout = QtWidgets.QVBoxLayout(self)

            label = QtWidgets.QLabel(
                "Review or edit the prompt that will be sent to OpenAI along with the transcript."
            )
            label.setWordWrap(True)
            layout.addWidget(label)

            self.text_edit = QtWidgets.QPlainTextEdit(default_prompt)
            self.text_edit.setMinimumHeight(200)
            layout.addWidget(self.text_edit)

            button_box = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
            )
            button_box.accepted.connect(self.accept)
            button_box.rejected.connect(self.reject)
            layout.addWidget(button_box)

        def prompt_text(self) -> str:
            return self.text_edit.toPlainText()

    class TranscriberWindow(QtWidgets.QMainWindow):
        def __init__(self, args: argparse.Namespace) -> None:
            super().__init__()
            self.args = args
            self.recorder = AudioRecorder(args.sample_rate, args.channels, args.device)
            self.is_recording = False
            self.elapsed_seconds = 0
            self.transcription_files: List[Path] = []
            self.worker_thread: Optional[QtCore.QThread] = None
            self.worker: Optional[TranscriptionWorker] = None
            self.clean_worker_thread: Optional[QtCore.QThread] = None
            self.clean_worker: Optional[OpenAICleanupWorker] = None
            self.current_segments: List[Dict] = []
            self.speaker_names: Dict[str, str] = {}
            self.current_saved_path: Optional[Path] = None

            self.setWindowTitle("Whisper Microphone Transcriber")
            self.resize(960, 600)
            self.setMinimumSize(760, 480)

            self.timer = QtCore.QTimer(self)
            self.timer.setInterval(1000)
            self.timer.timeout.connect(self.update_timer)

            self.refresh_timer = QtCore.QTimer(self)
            self.refresh_timer.setInterval(2000)
            self.refresh_timer.timeout.connect(self.sync_transcripts)

            self.setup_ui()
            self.refresh_file_list()
            self.refresh_timer.start()

        def setup_ui(self) -> None:
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)

            root_layout = QtWidgets.QVBoxLayout(central)

            header_layout = QtWidgets.QHBoxLayout()
            root_layout.addLayout(header_layout)

            self.record_button = QtWidgets.QPushButton("Transcribe")
            button_font = self.record_button.font()
            button_font.setPointSize(18)
            button_font.setBold(True)
            self.record_button.setFont(button_font)
            self.record_button.setMinimumHeight(60)
            self.record_button.clicked.connect(self.toggle_recording)
            header_layout.addWidget(self.record_button, stretch=0)

            info_layout = QtWidgets.QVBoxLayout()
            header_layout.addLayout(info_layout)

            status_row = QtWidgets.QHBoxLayout()
            status_title = QtWidgets.QLabel("Status:")
            status_title.setStyleSheet("font-weight: 600;")
            status_row.addWidget(status_title)
            self.status_label = QtWidgets.QLabel("Ready")
            status_row.addWidget(self.status_label)
            status_row.addStretch()
            info_layout.addLayout(status_row)

            timer_row = QtWidgets.QHBoxLayout()
            timer_title = QtWidgets.QLabel("Elapsed:")
            timer_title.setStyleSheet("font-weight: 600;")
            timer_row.addWidget(timer_title)
            self.timer_label = QtWidgets.QLabel("00:00")
            timer_row.addWidget(self.timer_label)
            timer_row.addStretch()
            info_layout.addLayout(timer_row)

            self.diarize_checkbox = QtWidgets.QCheckBox("Enable speaker diarization")
            self.diarize_checkbox.setChecked(self.args.diarize)
            info_layout.addWidget(self.diarize_checkbox)

            button_row = QtWidgets.QHBoxLayout()
            info_layout.addLayout(button_row)

            self.rename_button = QtWidgets.QPushButton("Rename Speakers")
            self.rename_button.clicked.connect(self.rename_speakers)
            button_row.addWidget(self.rename_button)

            self.clean_button = QtWidgets.QPushButton("Clean with OpenAI")
            self.clean_button.setEnabled(False)
            self.clean_button.clicked.connect(self.clean_with_openai)
            button_row.addWidget(self.clean_button)

            header_layout.addStretch()

            body_layout = QtWidgets.QHBoxLayout()
            root_layout.addLayout(body_layout, stretch=1)

            self.transcript_edit = QtWidgets.QPlainTextEdit()
            self.transcript_edit.setReadOnly(True)
            body_layout.addWidget(self.transcript_edit, stretch=2)

            sidebar_layout = QtWidgets.QVBoxLayout()
            body_layout.addLayout(sidebar_layout, stretch=1)

            list_title = QtWidgets.QLabel("Saved Transcriptions")
            list_title.setStyleSheet("font-weight: 600;")
            sidebar_layout.addWidget(list_title)

            self.list_widget = QtWidgets.QListWidget()
            self.list_widget.setSpacing(2)
            self.list_widget.currentRowChanged.connect(self.on_list_selected)
            sidebar_layout.addWidget(self.list_widget, stretch=1)

            folder_label = QtWidgets.QLabel(f"Folder:\n{TRANSCRIPT_DIR}")
            folder_label.setWordWrap(True)
            sidebar_layout.addWidget(folder_label)

        def toggle_recording(self) -> None:
            if self.is_recording:
                self.stop_recording()
            else:
                self.start_recording()

        def start_recording(self) -> None:
            try:
                self.recorder.start()
            except Exception as exc:  # noqa: BLE001
                QtWidgets.QMessageBox.critical(self, "Recording Error", str(exc))
                return

            self.is_recording = True
            self.elapsed_seconds = 0
            self.timer_label.setText("00:00")
            self.record_button.setText("Stop")
            self.record_button.setEnabled(True)
            self.status_label.setText("Recording… press Stop to transcribe")
            self.timer.start()
            self.clean_button.setEnabled(False)

        def stop_recording(self) -> None:
            self.record_button.setEnabled(False)
            audio = self.recorder.stop()
            self.is_recording = False
            self.timer.stop()
            self.status_label.setText("Processing transcription…")
            self.start_worker(audio)

        def start_worker(self, audio: np.ndarray) -> None:
            diarize = self.diarize_checkbox.isChecked()
            token = self.args.diarization_token or os.getenv("PYANNOTE_AUTH_TOKEN")
            if diarize and not token:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Speaker Diarization",
                    "No Hugging Face token provided (PYANNOTE_AUTH_TOKEN). Continuing without diarization.",
                )
                diarize = False

            worker = TranscriptionWorker(audio, self.args.sample_rate, self.args, diarize, token, {})
            thread = QtCore.QThread(self)
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.finished.connect(self.handle_transcription_finished)
            worker.finished.connect(worker.deleteLater)
            worker.finished.connect(thread.quit)
            thread.finished.connect(thread.deleteLater)
            thread.start()
            self.worker_thread = thread
            self.worker = worker

        def handle_transcription_finished(self, result: object, error: object) -> None:
            self.record_button.setText("Transcribe")
            self.record_button.setEnabled(True)

            if self.worker_thread is not None:
                self.worker_thread = None
            self.worker = None

            if error:
                self.status_label.setText("Transcription failed")
                QtWidgets.QMessageBox.critical(self, "Transcription Error", str(error))
                self.update_clean_button_state()
                return

            if not isinstance(result, dict):
                result = {}

            formatted_text = result.get("formatted_text") or result.get("raw_text", "")
            self.transcript_edit.setPlainText(formatted_text)
            self.status_label.setText("Transcription complete")
            self.timer_label.setText("00:00")

            saved_path_str = result.get("saved_path") or ""
            self.current_saved_path = Path(saved_path_str) if saved_path_str else None

            segments = result.get("segments") or []
            if result.get("diarized") and segments:
                unique_speakers = sorted({seg.get("speaker") for seg in segments if seg.get("speaker")})
                self.speaker_names = {speaker: speaker for speaker in unique_speakers}
            else:
                segments = []
                self.speaker_names = {}
            self.current_segments = segments

            self.refresh_file_list()
            if self.current_saved_path:
                self.select_file(self.current_saved_path)
            elif not segments:
                self.transcript_edit.setPlainText(formatted_text)

            self.update_clean_button_state()

        def update_timer(self) -> None:
            if not self.is_recording:
                return
            self.elapsed_seconds += 1
            minutes, seconds = divmod(self.elapsed_seconds, 60)
            self.timer_label.setText(f"{minutes:02d}:{seconds:02d}")

        def gather_files(self) -> List[Path]:
            TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
            files = [p.resolve() for p in TRANSCRIPT_DIR.glob("*.txt") if p.is_file()]
            files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            return files

        def refresh_file_list(self, files: Optional[List[Path]] = None) -> None:
            if files is None:
                files = self.gather_files()

            self.transcription_files = files
            self.list_widget.blockSignals(True)
            self.list_widget.clear()

            if not files:
                self.list_widget.addItem("(No transcriptions found)")
                self.list_widget.setEnabled(False)
            else:
                self.list_widget.setEnabled(True)
                for path in files:
                    modified = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                    item = QtWidgets.QListWidgetItem()
                    container = QtWidgets.QWidget()
                    layout = QtWidgets.QHBoxLayout(container)
                    layout.setContentsMargins(8, 2, 2, 2)
                    label = QtWidgets.QLabel(f"{modified}  {path.name}")
                    layout.addWidget(label)
                    layout.addStretch()
                    delete_button = QtWidgets.QToolButton()
                    delete_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_TrashIcon))
                    delete_button.setToolTip("Delete transcript")
                    delete_button.setCursor(QtCore.Qt.PointingHandCursor)
                    delete_button.clicked.connect(lambda _=False, p=path: self.delete_transcript(p))
                    layout.addWidget(delete_button)
                    container.setLayout(layout)
                    item.setSizeHint(container.sizeHint())
                    self.list_widget.addItem(item)
                    self.list_widget.setItemWidget(item, container)
            self.list_widget.blockSignals(False)
            self.update_clean_button_state()

        def sync_transcripts(self) -> None:
            current = self.gather_files()
            if current != self.transcription_files:
                self.refresh_file_list(current)

        def on_list_selected(self, row: int) -> None:
            if row < 0 or row >= len(self.transcription_files):
                return
            path = self.transcription_files[row]
            keep_segments = (
                self.current_saved_path is not None
                and path.resolve() == self.current_saved_path.resolve()
                and bool(self.current_segments)
            )

            self.display_file(path)
            self.current_saved_path = path

            if not keep_segments:
                self.current_segments = []
                self.speaker_names = {}
            self.update_clean_button_state()

        def display_file(self, path: Path) -> None:
            try:
                content = path.read_text(encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                QtWidgets.QMessageBox.critical(self, "File Error", f"Unable to read {path.name}: {exc}")
                return
            self.transcript_edit.setPlainText(content)
            self.status_label.setText(f"Showing {path.name}")
            self.update_clean_button_state()

        def select_file(self, path: Path) -> None:
            if path not in self.transcription_files:
                self.refresh_file_list()
                if path not in self.transcription_files:
                    return
            row = self.transcription_files.index(path)
            self.list_widget.setCurrentRow(row)
            self.display_file(path)
            self.update_clean_button_state()

        def delete_transcript(self, path: Path) -> None:
            if not path.exists():
                self.refresh_file_list()
                return

            reply = QtWidgets.QMessageBox.question(
                self,
                "Delete Transcript",
                f"Delete '{path.name}'? This cannot be undone.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return

            try:
                path.unlink()
            except OSError as exc:
                QtWidgets.QMessageBox.warning(self, "Delete Failed", f"Could not delete file: {exc}")
                return

            if self.current_saved_path and self.current_saved_path.resolve() == path.resolve():
                self.current_saved_path = None
                self.transcript_edit.clear()
                self.status_label.setText("Ready")
                self.current_segments = []
                self.speaker_names = {}

            self.refresh_file_list()
            self.update_clean_button_state()

        def rename_speakers(self) -> None:
            if not self.current_segments or not self.speaker_names:
                QtWidgets.QMessageBox.information(
                    self,
                    "No Speakers",
                    "There are no diarized speakers available to rename yet.",
                )
                return

            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle("Rename Speakers")
            form_layout = QtWidgets.QFormLayout(dialog)
            edits: Dict[str, QtWidgets.QLineEdit] = {}
            for speaker, current_name in self.speaker_names.items():
                line = QtWidgets.QLineEdit(current_name)
                edits[speaker] = line
                form_layout.addRow(f"{speaker}:", line)

            buttons = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
            )
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            form_layout.addWidget(buttons)

            if dialog.exec() == QtWidgets.QDialog.Accepted:
                for speaker, line in edits.items():
                    name = line.text().strip() or speaker
                    self.speaker_names[speaker] = name
                self.apply_speaker_names()

        def apply_speaker_names(self) -> None:
            if not self.current_segments:
                return
            formatted = format_transcript_segments(self.current_segments, self.speaker_names)
            self.transcript_edit.setPlainText(formatted)
            if self.current_saved_path:
                try:
                    self.current_saved_path.write_text(formatted + "\n", encoding="utf-8")
                except OSError as exc:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Save Error",
                        f"Could not update transcript file: {exc}",
                    )
            self.update_clean_button_state()

        def clean_with_openai(self) -> None:
            transcript = self.transcript_edit.toPlainText().strip()
            if not transcript:
                QtWidgets.QMessageBox.information(self, "Nothing to clean", "There is no transcript to send to OpenAI.")
                return

            if not os.getenv("OPENAI_API_KEY"):
                QtWidgets.QMessageBox.warning(
                    self,
                    "Missing API Key",
                    "OPENAI_API_KEY is not set. Add it to your .env file before using OpenAI cleanup.",
                )
                return

            prompt_dialog = PromptDialog(self, DEFAULT_CLEAN_PROMPT)
            if prompt_dialog.exec() != QtWidgets.QDialog.Accepted:
                return

            prompt_text = prompt_dialog.prompt_text().strip() or DEFAULT_CLEAN_PROMPT

            worker = OpenAICleanupWorker(transcript, prompt_text, DEFAULT_OPENAI_MODEL)
            thread = QtCore.QThread(self)
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.finished.connect(self.handle_cleanup_finished)
            worker.finished.connect(worker.deleteLater)
            worker.finished.connect(thread.quit)
            thread.finished.connect(thread.deleteLater)
            thread.start()
            self.clean_worker_thread = thread
            self.clean_worker = worker
            self.clean_button.setEnabled(False)
            self.status_label.setText("Cleaning transcript with OpenAI…")

        def handle_cleanup_finished(self, cleaned_text: object, error: object) -> None:
            if self.clean_worker_thread is not None:
                self.clean_worker_thread = None
            self.clean_worker = None

            if error:
                self.status_label.setText("OpenAI cleanup failed")
                QtWidgets.QMessageBox.critical(self, "OpenAI Error", str(error))
                self.update_clean_button_state()
                return

            if not isinstance(cleaned_text, str) or not cleaned_text.strip():
                self.status_label.setText("OpenAI cleanup returned no text")
                QtWidgets.QMessageBox.warning(
                    self, "OpenAI Cleanup", "OpenAI did not return any cleaned transcript."
                )
                self.update_clean_button_state()
                return

            cleaned_text = cleaned_text.strip()
            self.transcript_edit.setPlainText(cleaned_text)
            self.status_label.setText("Transcript cleaned with OpenAI")

            if self.current_saved_path:
                try:
                    self.current_saved_path.write_text(cleaned_text + "\n", encoding="utf-8")
                except OSError as exc:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Save Error",
                        f"Could not update transcript file: {exc}",
                    )

            self.current_segments = []
            self.speaker_names = {}
            self.update_clean_button_state()

        def update_clean_button_state(self) -> None:
            has_text = bool(self.transcript_edit.toPlainText().strip())
            has_key = bool(os.getenv("OPENAI_API_KEY"))
            self.clean_button.setEnabled(has_text and has_key)

        def closeEvent(self, event: QtCore.QEvent) -> None:  # noqa: D401
            if self.is_recording:
                self.recorder.stop()
            self.timer.stop()
            self.refresh_timer.stop()
            if self.worker_thread is not None:
                self.worker_thread.quit()
                self.worker_thread.wait(2000)
            if self.clean_worker_thread is not None:
                self.clean_worker_thread.quit()
                self.clean_worker_thread.wait(2000)
            self.worker = None
            self.clean_worker = None
            super().closeEvent(event)

    qt_app = QtWidgets.QApplication(sys.argv)
    window = TranscriberWindow(args)
    window.show()
    qt_app.exec()


def prompt_for_cleanup(default_prompt: str) -> str:
    print("\nDefault OpenAI cleanup prompt:\n")
    print(default_prompt)
    print("\nPress Enter to use this prompt, or type 'edit' to provide a custom one.")
    response = input("> ").strip().lower()
    if response in {"", "y", "yes"}:
        return default_prompt
    if response not in {"edit", "e"}:
        return default_prompt

    print("Enter your prompt. Submit with Ctrl-D (Ctrl-Z on Windows) when finished.\n")
    lines: List[str] = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        pass
    custom = "\n".join(lines).strip()
    return custom or default_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record audio from the microphone and transcribe it using Whisper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--duration", type=float, default=5.0, help="Recording duration in seconds (CLI only)")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Sample rate for recording")
    parser.add_argument("--channels", type=int, default=1, help="Number of input channels")
    parser.add_argument("--device", type=int, default=None, help="Sounddevice input device index")
    parser.add_argument("--model", type=str, default="base", help="Whisper model size to load")
    parser.add_argument("--language", type=str, default=None, help="Language code hint for transcription")
    parser.add_argument("--save-audio", type=Path, default=None, help="Optional path to save the recorded WAV audio (CLI only)")
    parser.add_argument("--whisper-device", type=str, default=None, help="Device to run Whisper on (cpu, cuda, etc.)")
    parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization using Pyannote models")
    parser.add_argument("--diarization-token", type=str, default=None, help="Hugging Face token for the diarization model")
    parser.add_argument(
        "--speaker-label",
        action="append",
        default=[],
        help="Map diarized speaker IDs to names (format SPEAKER_00=Alice). Can be provided multiple times.",
    )
    parser.add_argument("--openai-clean", action="store_true", help="Send transcript to OpenAI for cleanup after transcription")
    parser.add_argument("--openai-model", type=str, default=DEFAULT_OPENAI_MODEL, help="OpenAI model to use for cleanup")
    parser.add_argument("--list-devices", action="store_true", help="List available audio input devices and exit")
    parser.add_argument("--ui", action="store_true", help="Launch the graphical interface with start/stop controls")
    parser.add_argument("--openai-clean-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--openai-clean-input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--openai-clean-output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.openai_clean_worker:
        if not args.openai_clean_input or not args.openai_clean_output:
            print("Missing input/output paths for OpenAI cleanup worker", file=sys.stderr)
            sys.exit(1)
        run_openai_cleanup_worker(args.openai_clean_input, args.openai_clean_output)
        return

    if args.list_devices:
        list_audio_devices()
        return

    if args.ui:
        launch_ui(args)
        return

    try:
        audio = record_audio(
            duration=args.duration,
            sample_rate=args.sample_rate,
            channels=args.channels,
            device=args.device,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error while recording audio: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.save_audio:
        audio_path = Path(args.save_audio).expanduser().resolve()
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        write_wav(audio, args.sample_rate, audio_path)
    else:
        temp_file = tempfile.NamedTemporaryFile(prefix="whisper_recording_", suffix=".wav", delete=False)
        audio_path = Path(temp_file.name)
        temp_file.close()
        try:
            write_wav(audio, args.sample_rate, audio_path)
        except Exception:  # noqa: BLE001
            audio_path.unlink(missing_ok=True)
            raise

    try:
        result = transcribe_pipeline(
            audio_path,
            args.model,
            args.language,
            args.whisper_device,
            args.diarize,
            args.diarization_token,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Transcription failed: {exc}", file=sys.stderr)
        if not args.save_audio and audio_path.exists():
            audio_path.unlink(missing_ok=True)
        sys.exit(1)
    finally:
        if not args.save_audio and audio_path.exists():
            audio_path.unlink(missing_ok=True)

    final_text = result.get("raw_text", "")
    segments = result.get("segments") or []

    if args.diarize and result.get("diarized") and segments:
        try:
            speaker_map = parse_speaker_labels(args.speaker_label)
        except ValueError as exc:
            print(f"{exc}", file=sys.stderr)
            sys.exit(1)
        if not speaker_map:
            print(
                "Detected speakers: " + ", ".join(sorted({seg['speaker'] for seg in segments if seg.get('speaker')}))
            )
        final_text = format_transcript_segments(segments, speaker_map)
    else:
        final_text = result.get("raw_text", "")

    print("\nTranscription:\n" + final_text)

    if args.openai_clean:
        try:
            prompt_text = prompt_for_cleanup(DEFAULT_CLEAN_PROMPT)
            cleaned_text = clean_transcript_with_openai(final_text, prompt_text, args.openai_model)
        except Exception as exc:  # noqa: BLE001
            print(f"OpenAI cleanup failed: {exc}", file=sys.stderr)
        else:
            print("\nCleaned Transcript:\n" + cleaned_text)


if __name__ == "__main__":
    main()
