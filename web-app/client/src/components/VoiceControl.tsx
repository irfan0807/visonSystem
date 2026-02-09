/**
 * VoiceControl — Push-to-talk / toggle microphone control
 * Shows audio levels and recording state
 */

import React from 'react';
import { motion } from 'framer-motion';
import { Mic, MicOff, Volume2 } from 'lucide-react';

interface VoiceControlProps {
  isRecording: boolean;
  isPlaying: boolean;
  audioLevel: number;
  onToggleRecording: () => void;
  disabled: boolean;
}

export const VoiceControl: React.FC<VoiceControlProps> = ({
  isRecording,
  isPlaying,
  audioLevel,
  onToggleRecording,
  disabled,
}) => {
  return (
    <div className="flex flex-col items-center gap-3">
      {/* Audio level visualization */}
      <div className="flex items-end justify-center gap-[3px] h-8 min-w-[60px]">
        {isRecording ? (
          Array.from({ length: 7 }).map((_, i) => {
            const barHeight = Math.max(
              4,
              audioLevel * 32 * (1 - Math.abs(i - 3) * 0.15) + Math.random() * 4
            );
            return (
              <motion.div
                key={i}
                className="w-[3px] rounded-full bg-gradient-to-t from-neural-500 to-iris-400"
                animate={{ height: barHeight }}
                transition={{ duration: 0.1 }}
              />
            );
          })
        ) : isPlaying ? (
          Array.from({ length: 7 }).map((_, i) => (
            <div key={i} className="wave-bar" style={{ animationDelay: `${i * 0.08}s` }} />
          ))
        ) : (
          <div className="text-dark-600 text-xs">
            {disabled ? 'Connecting...' : 'Click to speak'}
          </div>
        )}
      </div>

      {/* Main mic button */}
      <motion.button
        onClick={onToggleRecording}
        disabled={disabled || isPlaying}
        whileTap={{ scale: 0.92 }}
        className={`
          relative w-16 h-16 rounded-full flex items-center justify-center transition-all duration-300
          ${
            isRecording
              ? 'bg-red-500/20 border-2 border-red-500 iris-glow-active'
              : isPlaying
                ? 'bg-green-500/20 border-2 border-green-500/50 cursor-not-allowed'
                : 'bg-dark-800 border-2 border-dark-600 hover:border-neural-500 hover:bg-neural-500/10'
          }
          disabled:opacity-40 disabled:cursor-not-allowed
        `}
      >
        {/* Pulse ring when recording */}
        {isRecording && (
          <motion.div
            className="absolute inset-0 rounded-full border-2 border-red-500"
            animate={{ scale: [1, 1.5], opacity: [0.5, 0] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          />
        )}

        {isPlaying ? (
          <Volume2 className="w-6 h-6 text-green-400" />
        ) : isRecording ? (
          <MicOff className="w-6 h-6 text-red-400" />
        ) : (
          <Mic className="w-6 h-6 text-dark-300" />
        )}
      </motion.button>

      {/* Status text */}
      <span
        className={`text-[10px] font-medium tracking-wider uppercase ${
          isRecording
            ? 'text-red-400'
            : isPlaying
              ? 'text-green-400'
              : 'text-dark-500'
        }`}
      >
        {isRecording ? 'Recording...' : isPlaying ? 'IRIS Speaking' : 'Push to Talk'}
      </span>
    </div>
  );
};
