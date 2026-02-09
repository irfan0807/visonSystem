/**
 * IRIS — Main Application
 * Intelligent Real-time Interactive System
 *
 * Combines webcam, microphone, chat, and neural network
 * into a unified conversational AI experience
 */

import React, { useState, useCallback, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { Settings, Maximize2, Minimize2, RotateCcw, Zap } from 'lucide-react';
import {
  IrisAvatar,
  VideoFeed,
  ChatPanel,
  VoiceControl,
  StatusIndicator,
  VisionPanel,
} from './components';
import type { ChatMessage } from './components';
import { useSocket, useWebcam, useMicrophone, useAudioPlayer } from './hooks';
import type { VisionAnalysis } from './hooks/useSocket';

type IrisState = 'idle' | 'listening' | 'thinking' | 'speaking';

export default function App() {
  // ─── Hooks ──────────────────────────────────────────────────────
  const socket = useSocket();
  const webcam = useWebcam();
  const mic = useMicrophone();
  const audioPlayer = useAudioPlayer();

  // ─── State ──────────────────────────────────────────────────────
  const [irisState, setIrisState] = useState<IrisState>('idle');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [visionAnalysis, setVisionAnalysis] = useState<VisionAnalysis | null>(null);
  const [showVisionOverlay, setShowVisionOverlay] = useState(true);
  const [isThinking, setIsThinking] = useState(false);
  const [expanded, setExpanded] = useState<'video' | 'chat' | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Refs for intervals
  const frameIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const visionIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ─── Socket Event Handlers ──────────────────────────────────────

  // Handle audio responses (TTS from server)
  useEffect(() => {
    socket.onAudioResponse((data) => {
      setIrisState('speaking');
      setIsThinking(false);

      // Add assistant message
      setMessages((prev) => [
        ...prev,
        {
          id: `msg_${Date.now()}`,
          role: 'assistant',
          content: data.text,
          timestamp: Date.now(),
        },
      ]);

      // Play audio response
      audioPlayer.playAudio(data.audio, data.format).then(() => {
        setIrisState('idle');
      });
    });
  }, [socket, audioPlayer]);

  // Handle chat stream responses
  useEffect(() => {
    let streamingId = '';
    let streamContent = '';

    socket.onChatStream((data) => {
      if (data.id !== streamingId) {
        // New message stream
        streamingId = data.id;
        streamContent = data.chunk;
        setMessages((prev) => [
          ...prev,
          {
            id: data.id,
            role: 'assistant',
            content: data.chunk,
            timestamp: Date.now(),
            isStreaming: true,
          },
        ]);
        setIsThinking(false);
        setIrisState('speaking');
      } else {
        // Continue streaming
        streamContent += data.chunk;
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === data.id
              ? { ...msg, content: streamContent, isStreaming: !data.done }
              : msg
          )
        );
      }

      if (data.done) {
        setIrisState('idle');
        streamingId = '';
        streamContent = '';
      }
    });
  }, [socket]);

  // Handle vision analysis updates
  useEffect(() => {
    socket.onVision((data) => {
      setVisionAnalysis(data);
      setIsAnalyzing(false);
    });
  }, [socket]);

  // ─── Frame Streaming ────────────────────────────────────────────

  // Send video frames to server periodically
  useEffect(() => {
    if (webcam.isActive && socket.connected) {
      // Send frames every 500ms
      frameIntervalRef.current = setInterval(() => {
        const frame = webcam.captureFrame();
        if (frame) {
          socket.sendFrame(frame);
        }
      }, 500);

      // Request vision analysis every 3 seconds
      visionIntervalRef.current = setInterval(() => {
        const frame = webcam.captureFrame();
        if (frame) {
          setIsAnalyzing(true);
          socket.requestVisionAnalysis(frame);
        }
      }, 3000);
    }

    return () => {
      if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);
      if (visionIntervalRef.current) clearInterval(visionIntervalRef.current);
    };
  }, [webcam.isActive, socket.connected, webcam, socket]);

  // ─── Voice Interaction ──────────────────────────────────────────

  const handleToggleRecording = useCallback(async () => {
    if (mic.isRecording) {
      // Stop recording and send to server
      const audioBlob = await mic.stopRecording();
      if (audioBlob && audioBlob.size > 0) {
        setIrisState('thinking');
        setIsThinking(true);

        // Add user message placeholder
        setMessages((prev) => [
          ...prev,
          {
            id: `user_${Date.now()}`,
            role: 'user',
            content: '🎤 [Voice message]',
            timestamp: Date.now(),
          },
        ]);

        // Convert blob to base64 and send via socket
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64 = (reader.result as string).split(',')[1];
          if (base64) {
            socket.sendAudio(base64, 'webm');
          }
        };
        reader.readAsDataURL(audioBlob);
      }
    } else {
      // Start recording
      setIrisState('listening');
      await mic.startRecording();
    }
  }, [mic, socket]);

  // ─── Text Chat ──────────────────────────────────────────────────

  const handleSendMessage = useCallback(
    (text: string) => {
      setMessages((prev) => [
        ...prev,
        {
          id: `user_${Date.now()}`,
          role: 'user',
          content: text,
          timestamp: Date.now(),
        },
      ]);

      setIsThinking(true);
      setIrisState('thinking');

      // Send via socket for streaming response
      socket.sendChatMessage(text, webcam.isActive);
    },
    [socket, webcam.isActive]
  );

  // ─── Camera Toggle ──────────────────────────────────────────────

  const handleToggleCamera = useCallback(() => {
    if (webcam.isActive) {
      webcam.stop();
      setVisionAnalysis(null);
    } else {
      webcam.start();
    }
  }, [webcam]);

  // ─── Reset Conversation ─────────────────────────────────────────

  const handleReset = useCallback(() => {
    setMessages([]);
    setVisionAnalysis(null);
    setIrisState('idle');
    setIsThinking(false);
  }, []);

  // ─── Render ─────────────────────────────────────────────────────

  return (
    <div className="h-screen flex flex-col neural-bg">
      {/* ═══ Header ═══ */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-dark-800/80 glass">
        <div className="flex items-center gap-3">
          <IrisAvatar state={irisState} size="sm" audioLevel={mic.audioLevel} />
          <div>
            <h1 className="text-sm font-bold tracking-wide">
              <span className="text-neural-400">IRIS</span>
              <span className="text-dark-400 font-normal ml-2 text-xs">
                Intelligent Real-time Interactive System
              </span>
            </h1>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <StatusIndicator
            connected={socket.connected}
            cameraActive={webcam.isActive}
            micActive={mic.isRecording}
            aiThinking={isThinking}
          />

          <div className="w-px h-5 bg-dark-700" />

          <button
            onClick={handleReset}
            className="p-2 text-dark-500 hover:text-dark-300 hover:bg-dark-800 rounded-lg transition-colors"
            title="Reset conversation"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* ═══ Main Content ═══ */}
      <main className="flex-1 flex overflow-hidden">
        {/* Left Panel — Video Feed */}
        <div
          className={`
            flex flex-col border-r border-dark-800/80 transition-all duration-300
            ${expanded === 'chat' ? 'w-0 opacity-0' : expanded === 'video' ? 'flex-1' : 'w-[45%]'}
          `}
        >
          {/* Video */}
          <div className="flex-1 p-4 relative">
            <VideoFeed
              videoRef={webcam.videoRef}
              canvasRef={webcam.canvasRef}
              isActive={webcam.isActive}
              visionAnalysis={visionAnalysis}
              showOverlay={showVisionOverlay}
              onToggleCamera={handleToggleCamera}
              onToggleOverlay={() => setShowVisionOverlay(!showVisionOverlay)}
            />
          </div>

          {/* Voice Control Bar */}
          <div className="px-4 py-4 border-t border-dark-800/80 flex items-center justify-center">
            <VoiceControl
              isRecording={mic.isRecording}
              isPlaying={audioPlayer.isPlaying}
              audioLevel={mic.audioLevel}
              onToggleRecording={handleToggleRecording}
              disabled={!socket.connected}
            />
          </div>
        </div>

        {/* Right Panel — Chat + Vision */}
        <div
          className={`
            flex flex-col transition-all duration-300
            ${expanded === 'video' ? 'w-0 opacity-0' : expanded === 'chat' ? 'flex-1' : 'w-[55%]'}
          `}
        >
          {/* Top: Chat */}
          <div className="flex-1 flex overflow-hidden">
            <div className="flex-1 flex flex-col">
              <ChatPanel
                messages={messages}
                onSendMessage={handleSendMessage}
                isThinking={isThinking}
                disabled={!socket.connected}
              />
            </div>

            {/* Side: Vision Panel */}
            <div className="w-72 border-l border-dark-800/80 hidden lg:flex flex-col">
              <VisionPanel analysis={visionAnalysis} isAnalyzing={isAnalyzing} />
            </div>
          </div>
        </div>
      </main>

      {/* ═══ Footer ═══ */}
      <footer className="flex items-center justify-between px-6 py-2 border-t border-dark-800/80 glass">
        <div className="flex items-center gap-2">
          <Zap className="w-3 h-3 text-neural-500" />
          <span className="text-[10px] text-dark-500">
            Powered by GPT-4 Vision • Whisper STT • Neural TTS
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-dark-600">
            {webcam.isActive ? 'Camera: 30fps' : 'Camera: Off'}
          </span>
          <span className="text-[10px] text-dark-600">•</span>
          <span className="text-[10px] text-dark-600">
            {socket.connected ? 'WebSocket: Connected' : 'WebSocket: Disconnected'}
          </span>
        </div>
      </footer>
    </div>
  );
}
