# Whisper Microphone Transcriber PRD

## 1. Product Overview
- Provide a cross-platform tool (CLI + optional desktop UI) that records microphone audio on demand and produces text transcripts locally using an OpenAI Whisper model installed on the machine.
- Deliver lightweight controls for recording duration, device selection, audio persistence, and Whisper configuration to suit developers and power users.
- Offer a one-click desktop interface for casual or non-technical users, with automatic transcript archiving.
- Avoid reliance on hosted APIs or network access so the tool remains private and usable offline after initial setup.

## 2. Goals & Non-Goals
### Goals
- Let users verify microphone availability and select the correct input device.
- Capture short- to medium-length recordings (≤ 5 minutes) with configurable duration, sample rate, and channels.
- Run Whisper inference locally with user-selectable model sizes and language hints.
- Offer optional audio file retention for debugging or re-use.
- Provide a “big button” UI to start/stop recordings, show live status, and display the most recent transcript.
- Persist transcripts to `~/Documents/transcriptions/<YYMMDDHHMMSS>.txt` and surface a reverse-chronological list in the UI.
- Surface actionable error messages for device, dependency, or transcription failures.
- Provide tools to rename diarized speakers after transcription so saved transcripts reflect real names.

### Non-Goals
- Building additional platform-specific GUIs beyond the bundled PySide6 desktop app.
- Real-time streaming transcription or live captioning.
- Automatic noise suppression, voice activity detection (VAD), or post-processing beyond Whisper defaults.
- Cloud-based transcription or remote storage of audio/text.

## 3. User Personas & Stories
- **Developer / Researcher** working on speech-enabled prototypes who needs quick offline transcripts.
  - As a developer, I want to list available devices so I can pick the right microphone.
  - As a developer, I want to record a short note and immediately see the transcript in my terminal.
  - As a developer, I want to save the audio file for later evaluation or model fine-tuning.
- **Content Creator / Journalist** capturing interviews or ideas on a laptop.
  - As a journalist, I want to transcribe an impromptu voice memo without uploading sensitive audio to external services.

## 4. Functional Requirements
- `--list-devices` flag prints microphone-capable devices with indices and exits.
- Recording command accepts:
  - `--duration` (float seconds, default 5) must be positive.
  - `--sample-rate` (int, default 16000) validated against device.
  - `--channels` (int, default 1) validated against device.
  - `--device` (int, optional) selects a specific device index.
- Audio capture uses `sounddevice` to record and stores data as 32-bit float.
- Recorded audio is written to WAV at the requested sample rate.
  - Temporary file automatically deleted unless `--save-audio PATH` is provided.
- Local transcription:
  - `--model` specifies Whisper checkpoint (default `base`).
  - `--language` provides an optional BCP-47 language hint.
  - `--whisper-device` selects compute target (`cpu`, `cuda`, etc.).
  - Failures produce clear stderr messages and exit with non-zero status.
- Terminal output displays the transcription block once complete.
- UI mode (`--ui`) launches a PySide6 desktop window with:
  - A large toggle button to start/stop recording using the default audio settings (configurable via CLI flags).
  - Status label reflecting recording/transcription progress and errors plus an elapsed timer.
  - Viewer showing the latest or selected transcript text.
  - List of saved transcript files ordered by newest first, updating after each transcription.
  - Optional speaker diarization (requires Hugging Face token) and a rename dialog to relabel speakers post-transcription.
  - Ability to send transcripts to OpenAI with an editable prompt, updating on-screen and persisted text once the response returns.

## 5. UX & Interaction Flow
1. User optionally runs `python app.py --list-devices` to gather device indices.
2. CLI path: user runs `python app.py [options]`.
   - CLI validates inputs, checks microphone access, and records audio while showing progress messages.
   - CLI writes WAV data, loads Whisper, transcribes, and prints "Transcription:" with resulting text.
   - Temporary files are cleaned up automatically unless explicitly saved.
3. UI path: user runs `python app.py --ui`.
   - User presses **Transcribe** to begin capture; the button toggles to **Stop**.
   - App records until the user stops, then performs transcription in the background.
   - Transcribed text appears in the viewer; a timestamped `.txt` file is written to the transcripts directory.
   - The saved files list refreshes automatically, allowing quick browsing of prior sessions.

## 6. Technical Considerations
- Dependencies: `sounddevice`, `soundfile`, `numpy`, `openai-whisper`, `ffmpeg`, PyTorch (transitive), `PySide6`, `python-dotenv` (to load local tokens/keys), `openai`.
- Environment: Python 3.9+, microphone permissions, accessible `ffmpeg` executable on PATH.
- Performance: Whisper load time proportional to model size; models are cached per process to avoid redundant loads.
- Memory: Larger models require significant RAM/VRAM; default remains `base` to balance accuracy and speed.

## 7. Success Metrics
- Recording and transcription succeed for ≥ 90% of attempts on supported hardware with valid configuration.
- CLI `--list-devices` returns accurate device info on macOS and Linux.
- Users can produce a transcription in under 60 seconds when using the default `base` model on CPU for <= 30s audio.
- UI users complete a capture-to-transcript flow in ≤ 3 button clicks, with the transcript saved to disk automatically.

## 8. Risks & Mitigations
- **Microphone access denied**: Document OS permission steps; catch device errors with actionable messaging.
- **Missing dependencies (ffmpeg/PyTorch)**: Provide setup instructions in README and error guidance in CLI.
- **Large model performance**: Warn users via README; allow specifying alternative models.
- **Sandbox limitations (CI, remote environments)**: Encourage running on local machine; provide `--list-devices` to detect unsupported cases.

## 9. Open Questions
- Should we add an interactive mode that records until pressing a key?
- Do we need automated tests (e.g., mock audio input) to validate CLI/UI behavior?
- Should transcripts optionally save to additional formats (JSON) or sync to cloud storage?

---

# Implementation Plan (Current)

## Milestones & Deliverables
1. **MVP CLI (Complete)**
   - Device listing, configurable recording, transcription, basic error handling.
2. **Desktop UI (Complete)**
   - Implement PySide6 window with start/stop toggle, elapsed timer, and status updates. ✅
   - Persist transcripts to `~/Documents/transcriptions` and list them newest-first. ✅
   - Surface transcription results in-app. ✅
   - Integrate optional OpenAI cleanup workflow with customizable prompt. ✅
3. **Reliability & Automation (Planned)**
   - Add unit/integration tests with mocked audio input.
   - Automate dependency checks (ffmpeg detection) with improved messaging.
4. **Extensibility (Future Consideration)**
   - Support exporting transcripts to additional formats (JSON) or direct copy-to-clipboard.
   - Integrate optional noise suppression or VAD.

## Task Breakdown
- **Dependency Management**
  - Ensure README and setup scripts validate `ffmpeg` availability.
  - Provide script or documentation for installing required system packages per OS.
- **Recording UX**
  - Implement countdown display and optional audible alert when recording starts/ends (open).
  - Implement `--interactive` flag or hotkey support for CLI to stop on demand (open).
- **Transcription Output**
  - Add `--output` parameter to write transcript to file (text/JSON).
  - Display detected language/confidence when auto-detection is used (both CLI & UI).
- **Error Handling & Logging**
  - Differentiate between dependency, permission, and runtime errors.
  - Surface helpful guidance when Whisper model download or load fails.
- **Testing**
  - Add CLI smoke test using generated sine wave audio fixture.
  - Mock `sounddevice` input to validate error paths without hardware.
- **Packaging & Distribution** (future)
  - Provide optional `setup.py` or `pipx` entry point for easier installation.

## Status Tracking
- Current version delivers Milestones 1 and 2 (CLI + PySide6 desktop UI with timer, archive list, and viewer).
- Blocking issue: environment without `ffmpeg` prevents transcription (mitigated by README instructions; automation pending).
- Next focus: dependency validation (ensure `ffmpeg` error surfaced early) and optional transcript export/formatting.
