/**
 * WebSocket Handler
 * Real-time bidirectional communication for audio/video/chat
 */

import { Server as SocketServer, Socket } from 'socket.io';
import { Server as HttpServer } from 'http';
import { NeuralNetworkService } from '../services/NeuralNetworkService';
import type {
  ClientToServerEvents,
  ServerToClientEvents,
  SystemStatus,
  FrameData,
} from '../types';

export function setupWebSocket(
  httpServer: HttpServer,
  neuralNetwork: NeuralNetworkService,
  corsOrigin: string
): SocketServer {
  const io = new SocketServer<ClientToServerEvents, ServerToClientEvents>(httpServer, {
    cors: {
      origin: corsOrigin,
      methods: ['GET', 'POST'],
    },
    maxHttpBufferSize: 10e6, // 10MB for audio/video data
  });

  let activeConnections = 0;
  let lastActivity = Date.now();

  io.on('connection', (socket: Socket<ClientToServerEvents, ServerToClientEvents>) => {
    activeConnections++;
    console.log(`[WebSocket] Client connected: ${socket.id} (${activeConnections} active)`);

    // Send initial status
    emitStatus(socket);

    // ─── Audio Stream Handler ─────────────────────────────────────
    socket.on('audio:stream', async (data) => {
      lastActivity = Date.now();
      try {
        const audioBuffer = Buffer.from(data.audio, 'base64');
        const format = data.format || 'webm';

        // Get latest frame for multimodal processing
        const latestFrame = (socket as any).__latestFrame as FrameData | undefined;

        // Full multimodal pipeline: STT → GPT → TTS
        const result = await neuralNetwork.processMultimodal(
          audioBuffer,
          format,
          latestFrame
        );

        // Send back audio response + text
        socket.emit('audio:response', {
          audio: result.audioBuffer.toString('base64'),
          text: result.text,
          format: 'mp3',
        });

        // Send vision analysis if available
        if (result.visionAnalysis) {
          socket.emit('vision:analysis', result.visionAnalysis);
        }
      } catch (error) {
        console.error('[WebSocket] Audio processing error:', error);
        socket.emit('error', {
          message: 'Failed to process audio',
          code: 'AUDIO_PROCESSING_ERROR',
        });
      }
    });

    // ─── Video Frame Handler ──────────────────────────────────────
    socket.on('video:frame', async (data) => {
      lastActivity = Date.now();
      try {
        const frame: FrameData = {
          imageBase64: data.frame,
          timestamp: data.timestamp || Date.now(),
          width: 640,
          height: 480,
        };

        // Store latest frame on socket for multimodal use
        (socket as any).__latestFrame = frame;
      } catch (error) {
        console.error('[WebSocket] Frame processing error:', error);
      }
    });

    // ─── Vision Analysis Request ──────────────────────────────────
    socket.on('vision:analyze', async (data) => {
      lastActivity = Date.now();
      try {
        const frame: FrameData = {
          imageBase64: data.frame,
          timestamp: Date.now(),
          width: 640,
          height: 480,
        };

        const analysis = await neuralNetwork.analyzeFrame(frame);
        socket.emit('vision:analysis', analysis);
      } catch (error) {
        console.error('[WebSocket] Vision analysis error:', error);
        socket.emit('error', {
          message: 'Failed to analyze frame',
          code: 'VISION_ANALYSIS_ERROR',
        });
      }
    });

    // ─── Chat Message Handler ─────────────────────────────────────
    socket.on('chat:message', async (data) => {
      lastActivity = Date.now();
      try {
        const latestFrame = data.includeVision
          ? (socket as any).__latestFrame as FrameData | undefined
          : undefined;

        const messageId = `msg_${Date.now()}`;

        // Stream the response for real-time feel
        const stream = neuralNetwork.chatStream(
          data.text,
          neuralNetwork.getLastVisionAnalysis(),
          latestFrame
        );

        let fullText = '';
        for await (const chunk of stream) {
          fullText += chunk;
          socket.emit('chat:stream', {
            chunk,
            id: messageId,
            done: false,
          });
        }

        socket.emit('chat:stream', {
          chunk: '',
          id: messageId,
          done: true,
        });
      } catch (error) {
        console.error('[WebSocket] Chat error:', error);
        socket.emit('error', {
          message: 'Failed to process message',
          code: 'CHAT_ERROR',
        });
      }
    });

    // ─── System Configuration ─────────────────────────────────────
    socket.on('system:configure', (data) => {
      neuralNetwork.updateConfig(data);
      emitStatus(socket);
    });

    // ─── Disconnect Handler ───────────────────────────────────────
    socket.on('disconnect', () => {
      activeConnections--;
      console.log(`[WebSocket] Client disconnected: ${socket.id} (${activeConnections} active)`);
    });
  });

  function emitStatus(socket: Socket) {
    const status: SystemStatus = {
      connected: true,
      modelsLoaded: true,
      activeStreams: activeConnections,
      lastActivity,
      visionActive: true,
      audioActive: true,
      fps: 0,
      latencyMs: 0,
    };
    socket.emit('system:status', status);
  }

  return io;
}
