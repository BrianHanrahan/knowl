import argparse
import functools
import sys
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper


DEFAULT_SAMPLE_RATE = 16_000
TRANSCRIPT_DIR = Path.home() / "Documents" / "transcriptions"


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


def transcribe_audio(audio_path: Path, model_name: str, language: Optional[str], device: Optional[str]) -> str:
    """Transcribe audio using a cached Whisper model."""
    print("Transcribing audio...")
    model = load_whisper_model(model_name, device)
    result = model.transcribe(str(audio_path), language=language)
    return result.get("text", "").strip()


def save_transcription_text(text: str) -> Path:
    """Persist transcription text to the timestamped Documents/transcriptions directory."""
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    path = TRANSCRIPT_DIR / f"{timestamp}.txt"
    counter = 1
    while path.exists():
        path = TRANSCRIPT_DIR / f"{timestamp}_{counter}.txt"
        counter += 1
    path.write_text(text + "\n", encoding="utf-8")
    return path


class AudioRecorder:
    """Manage a start/stop microphone recording session."""

    def __init__(self, sample_rate: int, channels: int, device: Optional[int]):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self._stream: Optional[sd.InputStream] = None
        self._frames: list[np.ndarray] = []
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
        finished = QtCore.Signal(str, object, object)

        def __init__(self, audio: np.ndarray, sample_rate: int, args: argparse.Namespace):
            super().__init__()
            self.audio = audio
            self.sample_rate = sample_rate
            self.args = args

        @QtCore.Slot()
        def run(self) -> None:
            text = ""
            saved_path: Optional[Path] = None
            error: Optional[str] = None
            temp_path: Optional[Path] = None

            try:
                if self.audio.size == 0:
                    raise RuntimeError("No audio captured. Try recording again.")

                with tempfile.NamedTemporaryFile(prefix="whisper_recording_", suffix=".wav", delete=False) as tmp:
                    temp_path = Path(tmp.name)

                write_wav(self.audio, self.sample_rate, temp_path)
                text = transcribe_audio(temp_path, self.args.model, self.args.language, self.args.whisper_device)
                if not text:
                    text = "[No speech detected]"
                saved_path = save_transcription_text(text)
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)

            saved_path_str = str(saved_path) if saved_path else ""
            self.finished.emit(text, saved_path_str, error)

    class TranscriberWindow(QtWidgets.QMainWindow):
        def __init__(self, args: argparse.Namespace) -> None:
            super().__init__()
            self.args = args
            self.recorder = AudioRecorder(args.sample_rate, args.channels, args.device)
            self.is_recording = False
            self.elapsed_seconds = 0
            self.transcription_files: list[Path] = []
            self.worker_thread: Optional[QtCore.QThread] = None
            self.worker: Optional[TranscriptionWorker] = None

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

        def stop_recording(self) -> None:
            self.record_button.setEnabled(False)
            audio = self.recorder.stop()
            self.is_recording = False
            self.timer.stop()
            self.status_label.setText("Processing transcription…")
            self.start_worker(audio)

        def start_worker(self, audio: np.ndarray) -> None:
            worker = TranscriptionWorker(audio, self.args.sample_rate, self.args)
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

        def handle_transcription_finished(self, text: str, saved_path_str: object, error: object) -> None:
            self.record_button.setText("Transcribe")
            self.record_button.setEnabled(True)

            if error:
                self.status_label.setText("Transcription failed")
                QtWidgets.QMessageBox.critical(self, "Transcription Error", str(error))
            else:
                self.status_label.setText("Transcription complete")
                self.transcript_edit.setPlainText(text)
                self.timer_label.setText("00:00")
                self.refresh_file_list()

                if saved_path_str:
                    saved_path = Path(str(saved_path_str))
                    self.select_file(saved_path)

            if self.worker_thread is not None:
                self.worker_thread = None
            self.worker = None

        def update_timer(self) -> None:
            if not self.is_recording:
                return
            self.elapsed_seconds += 1
            minutes, seconds = divmod(self.elapsed_seconds, 60)
            self.timer_label.setText(f"{minutes:02d}:{seconds:02d}")

        def gather_files(self) -> list[Path]:
            TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
            files = [p.resolve() for p in TRANSCRIPT_DIR.glob("*.txt") if p.is_file()]
            files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            return files

        def refresh_file_list(self, files: Optional[list[Path]] = None) -> None:
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
                    self.list_widget.addItem(f"{modified}  {path.name}")
            self.list_widget.blockSignals(False)

        def sync_transcripts(self) -> None:
            current = self.gather_files()
            if current != self.transcription_files:
                self.refresh_file_list(current)

        def on_list_selected(self, row: int) -> None:
            if row < 0 or row >= len(self.transcription_files):
                return
            self.display_file(self.transcription_files[row])

        def display_file(self, path: Path) -> None:
            try:
                content = path.read_text(encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                QtWidgets.QMessageBox.critical(self, "File Error", f"Unable to read {path.name}: {exc}")
                return
            self.transcript_edit.setPlainText(content)
            self.status_label.setText(f"Showing {path.name}")

        def select_file(self, path: Path) -> None:
            if path not in self.transcription_files:
                return
            row = self.transcription_files.index(path)
            self.list_widget.setCurrentRow(row)
            self.display_file(path)

        def closeEvent(self, event: QtCore.QEvent) -> None:  # noqa: D401
            if self.is_recording:
                self.recorder.stop()
            self.timer.stop()
            self.refresh_timer.stop()
            if self.worker_thread is not None:
                self.worker_thread.quit()
                self.worker_thread.wait(2000)
            self.worker = None
            super().closeEvent(event)

    qt_app = QtWidgets.QApplication(sys.argv)
    window = TranscriberWindow(args)
    window.show()
    qt_app.exec()


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
    parser.add_argument("--list-devices", action="store_true", help="List available audio input devices and exit")
    parser.add_argument("--ui", action="store_true", help="Launch the graphical interface with start/stop controls")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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
        temp_path = Path(temp_file.name)
        temp_file.close()
        try:
            write_wav(audio, args.sample_rate, temp_path)
        except Exception:  # noqa: BLE001
            temp_path.unlink(missing_ok=True)
            raise
        audio_path = temp_path

    try:
        transcription = transcribe_audio(
            audio_path,
            args.model,
            args.language,
            args.whisper_device,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Transcription failed: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if not args.save_audio and audio_path.exists():
            audio_path.unlink(missing_ok=True)

    print("\nTranscription:\n" + transcription)


if __name__ == "__main__":
    main()
