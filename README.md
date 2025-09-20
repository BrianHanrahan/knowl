# Whisper Microphone Transcriber

This command-line tool captures audio from your microphone and transcribes it locally using [OpenAI Whisper](https://github.com/openai/whisper).

## Prerequisites

- Python 3.9+
- `ffmpeg` available on your PATH (required by Whisper)
- Working microphone input device
- `PySide6` (installed automatically via `requirements.txt`) for the desktop UI
- Optional: Hugging Face access token (`PYANNOTE_AUTH_TOKEN`) to enable speaker diarization (you can place this in `.env`)
- Optional: OpenAI API key (`OPENAI_API_KEY`) if you plan to run transcript cleanup

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Record 5 seconds of audio (default) and transcribe with the Whisper `base` model:

```bash
source .venv/bin/activate
python app.py
```

Useful options:

- `--duration`: Recording length in seconds, e.g. `--duration 10`
- `--model`: Whisper model name, e.g. `--model small`
- `--save-audio`: Keep the recorded WAV file, e.g. `--save-audio recordings/latest.wav`
- `--list-devices`: Show available microphone devices
- `--device`: Use a specific device index from `--list-devices`
- `--language`: Provide a language hint (e.g. `--language en`)
- `--whisper-device`: Pass `cuda` to use a GPU-enabled setup (if available)
- `--diarize`: Run speaker diarization (requires Hugging Face token)
- `--diarization-token`: Supply the Hugging Face token on the command line
- `--speaker-label`: Map diarized speaker IDs to names, e.g. `--speaker-label SPEAKER_00=Alice` (repeatable)
- `--openai-clean`: Send the finished transcript to OpenAI for cleanup (interactive prompt lets you adjust instructions)
- `--openai-model`: Choose which OpenAI model to use (default: `gpt-4o-mini`)
- `--ui`: Launch the graphical interface instead of running in the terminal

For example, to record 15 seconds using device #2 and keep the audio file:

```bash
python app.py --duration 15 --device 2 --save-audio recordings/sample.wav
```

### Graphical UI

Launch the PySide6 desktop UI:

```bash
python app.py --ui
```

- Press **Transcribe** to start recording; the button switches to **Stop** and the timer counts elapsed seconds.
- Press **Stop** to finish. The capture is transcribed locally and saved to `~/Documents/transcriptions/<YYMMDDHHMMSS>.txt`.
- Toggle **Enable speaker diarization** to obtain speaker-attributed transcripts (requires a Hugging Face token).
- Use **Rename Speakers** after diarization completes to replace `SPEAKER_00`-style labels with real names; the saved transcript updates automatically.
- Press **Clean with OpenAI** to review and customize the cleanup prompt before sending the transcript to OpenAI; the returned version is written back to disk.
- Use the trash can icon beside a transcript to delete it from disk.
- The transcript list (newest first) refreshes automatically. Selecting an entry updates the viewer with the stored text.

### Speaker Diarization

1. Set `PYANNOTE_AUTH_TOKEN` in your environment (or pass `--diarization-token <token>` on the command line). Tokens are available from [Hugging Face](https://huggingface.co/pyannote/speaker-diarization-3.1).
2. Enable diarization via `--diarize` on the CLI or the checkbox in the UI.
3. After transcription, assign friendly names either with repeated `--speaker-label SPEAKER_00=Alice` flags (CLI) or via the **Rename Speakers** button in the UI.
4. Updated names are written back to the stored transcript files.

### Transcript Cleanup with OpenAI

1. Add `OPENAI_API_KEY` to your `.env` file (or export it in your shell).
2. Run the CLI with `--openai-clean` or press **Clean with OpenAI** in the UI after transcription completes.
3. Review the suggested prompt; tweak it to match your editing needs before the request is sent.
4. The cleaned transcript replaces the on-screen text and is written back to the saved file.

## Troubleshooting

- If recording fails, confirm the device index and that no other application is using the microphone.
- Ensure `ffmpeg` is installed (`ffmpeg -version`).
- Loading larger models (e.g. `large`) requires more memory and time.
```
