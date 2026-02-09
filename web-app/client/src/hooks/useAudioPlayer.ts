/**
 * Audio playback hook — plays TTS audio responses
 */

import { useRef, useState, useCallback } from 'react';

export interface AudioPlayerControls {
  isPlaying: boolean;
  playAudio: (base64Audio: string, format?: string) => Promise<void>;
  stop: () => void;
}

export function useAudioPlayer(): AudioPlayerControls {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  const playAudio = useCallback(async (base64Audio: string, format: string = 'mp3') => {
    try {
      // Stop any currently playing audio
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }

      // Decode base64 to blob
      const byteCharacters = atob(base64Audio);
      const byteNumbers = new Uint8Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const blob = new Blob([byteNumbers], { type: `audio/${format}` });
      const url = URL.createObjectURL(blob);

      const audio = new Audio(url);
      audioRef.current = audio;

      audio.onplay = () => setIsPlaying(true);
      audio.onended = () => {
        setIsPlaying(false);
        URL.revokeObjectURL(url);
      };
      audio.onerror = () => {
        setIsPlaying(false);
        URL.revokeObjectURL(url);
      };

      await audio.play();
    } catch (err) {
      console.error('[AudioPlayer] Playback error:', err);
      setIsPlaying(false);
    }
  }, []);

  const stop = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
      setIsPlaying(false);
    }
  }, []);

  return { isPlaying, playAudio, stop };
}
