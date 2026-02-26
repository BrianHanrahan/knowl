import { useEffect, useRef } from "react";
import { useSpeechRecognition } from "../hooks/useSpeechRecognition";

interface Props {
  onTranscript: (text: string) => void;
  disabled?: boolean;
  className?: string;
}

export default function VoiceMicButton({ onTranscript, disabled, className }: Props) {
  const { isListening, transcript, interimText, startListening, stopListening, supported } =
    useSpeechRecognition();
  const prevTranscriptRef = useRef("");

  // Fire onTranscript whenever new final text arrives
  useEffect(() => {
    if (transcript && transcript !== prevTranscriptRef.current) {
      const newText = transcript.slice(prevTranscriptRef.current.length);
      if (newText) {
        onTranscript(newText);
      }
      prevTranscriptRef.current = transcript;
    }
  }, [transcript, onTranscript]);

  // Reset ref when not listening
  useEffect(() => {
    if (!isListening) {
      prevTranscriptRef.current = "";
    }
  }, [isListening]);

  if (!supported) return null;

  const toggle = () => {
    if (isListening) {
      stopListening();
    } else {
      startListening();
    }
  };

  return (
    <span className={`voice-mic-wrapper ${className || ""}`}>
      <button
        type="button"
        className={`voice-mic-btn ${isListening ? "listening" : ""}`}
        onClick={toggle}
        disabled={disabled}
        title={isListening ? "Stop listening" : "Voice input"}
      >
        <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
          <path d="M8 1a2.5 2.5 0 0 0-2.5 2.5v4a2.5 2.5 0 0 0 5 0v-4A2.5 2.5 0 0 0 8 1z" />
          <path d="M3.5 7.5a.5.5 0 0 1 1 0 3.5 3.5 0 0 0 7 0 .5.5 0 0 1 1 0 4.5 4.5 0 0 1-4 4.473V13.5h1.5a.5.5 0 0 1 0 1h-4a.5.5 0 0 1 0-1H6.5v-1.527A4.5 4.5 0 0 1 3.5 7.5z" />
        </svg>
      </button>
      {isListening && interimText && (
        <span className="voice-interim">{interimText}</span>
      )}
    </span>
  );
}
