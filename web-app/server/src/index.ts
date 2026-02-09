/**
 * Vision System Server — Entry Point
 * Express + WebSocket server with real-time neural network integration
 */

import dotenv from 'dotenv';
dotenv.config();

import express from 'express';
import cors from 'cors';
import { createServer } from 'http';
import { NeuralNetworkService } from './services/NeuralNetworkService';
import { setupWebSocket } from './websocket/socketHandler';
import { createApiRoutes } from './routes/api';

const PORT = parseInt(process.env.PORT || '3001');
const CORS_ORIGIN = process.env.CORS_ORIGIN || 'http://localhost:5173';

async function main() {
  // ─── Initialize Express ─────────────────────────────────────────
  const app = express();
  const httpServer = createServer(app);

  app.use(cors({ origin: CORS_ORIGIN }));
  app.use(express.json({ limit: '10mb' }));
  app.use(express.urlencoded({ extended: true, limit: '10mb' }));

  // ─── Initialize Neural Network Service ──────────────────────────
  const neuralNetwork = new NeuralNetworkService();

  console.log('╔══════════════════════════════════════════════════════════╗');
  console.log('║          🧠 IRIS — Vision System Neural Engine          ║');
  console.log('╠══════════════════════════════════════════════════════════╣');
  console.log(`║  GPT Model:     ${(process.env.GPT_MODEL || 'gpt-4o').padEnd(38)}║`);
  console.log(`║  Whisper Model:  ${(process.env.WHISPER_MODEL || 'whisper-1').padEnd(37)}║`);
  console.log(`║  TTS Model:     ${(process.env.TTS_MODEL || 'tts-1').padEnd(38)}║`);
  console.log(`║  TTS Voice:     ${(process.env.TTS_VOICE || 'nova').padEnd(38)}║`);
  console.log('╚══════════════════════════════════════════════════════════╝');

  // ─── Setup REST API Routes ──────────────────────────────────────
  app.use('/api', createApiRoutes(neuralNetwork));

  // ─── Setup WebSocket ────────────────────────────────────────────
  const io = setupWebSocket(httpServer, neuralNetwork, CORS_ORIGIN);

  // ─── Start Server ───────────────────────────────────────────────
  httpServer.listen(PORT, () => {
    console.log(`\n🚀 Server running on http://localhost:${PORT}`);
    console.log(`📡 WebSocket ready on ws://localhost:${PORT}`);
    console.log(`🌐 CORS enabled for: ${CORS_ORIGIN}\n`);
  });
}

main().catch((error) => {
  console.error('Fatal error starting server:', error);
  process.exit(1);
});
