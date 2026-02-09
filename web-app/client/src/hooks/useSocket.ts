/**
 * Socket.IO hook for real-time communication with the server
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';

const SOCKET_URL = import.meta.env.VITE_SOCKET_URL || 'http://localhost:3001';

export interface VisionAnalysis {
  description: string;
  objects: { label: string; confidence: number }[];
  emotions?: { emotion: string; confidence: number }[];
  scene: string;
  anomalies: string[];
  timestamp: number;
}

export interface SystemStatus {
  connected: boolean;
  modelsLoaded: boolean;
  activeStreams: number;
  lastActivity: number;
  visionActive: boolean;
  audioActive: boolean;
  fps: number;
  latencyMs: number;
}

export function useSocket() {
  const socketRef = useRef<Socket | null>(null);
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState<SystemStatus | null>(null);

  // Callback refs for event handlers
  const onAudioResponseRef = useRef<((data: { audio: string; text: string; format: string }) => void) | null>(null);
  const onChatStreamRef = useRef<((data: { chunk: string; id: string; done: boolean }) => void) | null>(null);
  const onVisionRef = useRef<((data: VisionAnalysis) => void) | null>(null);
  const onErrorRef = useRef<((data: { message: string; code: string }) => void) | null>(null);

  useEffect(() => {
    const socket = io(SOCKET_URL, {
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionAttempts: 10,
    });

    socketRef.current = socket;

    socket.on('connect', () => {
      console.log('[Socket] Connected:', socket.id);
      setConnected(true);
    });

    socket.on('disconnect', () => {
      console.log('[Socket] Disconnected');
      setConnected(false);
    });

    socket.on('system:status', (data: SystemStatus) => {
      setStatus(data);
    });

    socket.on('audio:response', (data) => {
      onAudioResponseRef.current?.(data);
    });

    socket.on('chat:stream', (data) => {
      onChatStreamRef.current?.(data);
    });

    socket.on('vision:analysis', (data) => {
      onVisionRef.current?.(data);
    });

    socket.on('error', (data) => {
      onErrorRef.current?.(data);
      console.error('[Socket] Error:', data);
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  const sendAudio = useCallback((audioBase64: string, format: string = 'webm') => {
    socketRef.current?.emit('audio:stream', { audio: audioBase64, format });
  }, []);

  const sendFrame = useCallback((frameBase64: string) => {
    socketRef.current?.emit('video:frame', {
      frame: frameBase64,
      timestamp: Date.now(),
    });
  }, []);

  const sendChatMessage = useCallback((text: string, includeVision: boolean = true) => {
    socketRef.current?.emit('chat:message', { text, includeVision });
  }, []);

  const requestVisionAnalysis = useCallback((frameBase64: string) => {
    socketRef.current?.emit('vision:analyze', { frame: frameBase64 });
  }, []);

  return {
    connected,
    status,
    sendAudio,
    sendFrame,
    sendChatMessage,
    requestVisionAnalysis,
    onAudioResponse: (cb: typeof onAudioResponseRef.current) => { onAudioResponseRef.current = cb; },
    onChatStream: (cb: typeof onChatStreamRef.current) => { onChatStreamRef.current = cb; },
    onVision: (cb: typeof onVisionRef.current) => { onVisionRef.current = cb; },
    onError: (cb: typeof onErrorRef.current) => { onErrorRef.current = cb; },
  };
}
