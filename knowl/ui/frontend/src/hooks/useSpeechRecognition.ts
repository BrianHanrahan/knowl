import { useState, useEffect, useRef, useCallback } from "react";

interface SpeechRecognitionEvent {
  results: SpeechRecognitionResultList;
  resultIndex: number;
}

interface SpeechRecognitionInstance extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: ((event: SpeechRecognitionEvent) => void) | null;
  onend: (() => void) | null;
  onerror: ((event: { error: string }) => void) | null;
}

declare global {
  interface Window {
    SpeechRecognition: new () => SpeechRecognitionInstance;
    webkitSpeechRecognition: new () => SpeechRecognitionInstance;
  }
}

export interface UseSpeechRecognitionReturn {
  isListening: boolean;
  transcript: string;
  interimText: string;
  startListening: () => void;
  stopListening: () => void;
  supported: boolean;
}

const SpeechRecognitionAPI =
  typeof window !== "undefined"
    ? window.SpeechRecognition || window.webkitSpeechRecognition
    : null;

export function useSpeechRecognition(): UseSpeechRecognitionReturn {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [interimText, setInterimText] = useState("");
  const recognitionRef = useRef<SpeechRecognitionInstance | null>(null);
  const wantListeningRef = useRef(false);

  const supported = !!SpeechRecognitionAPI;

  const stopListening = useCallback(() => {
    wantListeningRef.current = false;
    setIsListening(false);
    setInterimText("");
    recognitionRef.current?.stop();
  }, []);

  const startListening = useCallback(() => {
    if (!SpeechRecognitionAPI) return;

    // Reset transcript for new session
    setTranscript("");
    setInterimText("");
    wantListeningRef.current = true;
    setIsListening(true);

    const recognition = new SpeechRecognitionAPI();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      let interim = "";
      let final = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        if (result.isFinal) {
          final += result[0].transcript;
        } else {
          interim += result[0].transcript;
        }
      }
      if (final) {
        setTranscript((prev) => prev + final);
        setInterimText("");
      } else {
        setInterimText(interim);
      }
    };

    recognition.onend = () => {
      // Browser sometimes stops recognition on its own — restart if we still want it
      if (wantListeningRef.current) {
        try {
          recognition.start();
        } catch {
          // Already started or other error — ignore
          setIsListening(false);
          wantListeningRef.current = false;
        }
      } else {
        setIsListening(false);
      }
    };

    recognition.onerror = (event: { error: string }) => {
      if (event.error === "not-allowed" || event.error === "service-not-allowed") {
        wantListeningRef.current = false;
        setIsListening(false);
      }
      // "no-speech" and "aborted" are transient — onend will handle restart
    };

    recognitionRef.current = recognition;

    try {
      recognition.start();
    } catch {
      setIsListening(false);
      wantListeningRef.current = false;
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      wantListeningRef.current = false;
      recognitionRef.current?.abort();
    };
  }, []);

  return {
    isListening,
    transcript,
    interimText,
    startListening,
    stopListening,
    supported,
  };
}
