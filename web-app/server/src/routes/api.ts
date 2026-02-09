/**
 * Express REST API Routes
 * For non-real-time operations and configuration
 */

import { Router, Request, Response } from 'express';
import multer from 'multer';
import { NeuralNetworkService } from '../services/NeuralNetworkService';
import type { FrameData } from '../types';

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB
});

export function createApiRoutes(neuralNetwork: NeuralNetworkService): Router {
  const router = Router();

  // ─── Health Check ─────────────────────────────────────────────
  router.get('/health', (_req: Request, res: Response) => {
    res.json({
      status: 'ok',
      timestamp: Date.now(),
      services: {
        neuralNetwork: 'active',
        gptModel: process.env.GPT_MODEL || 'gpt-4o',
        ttsModel: process.env.TTS_MODEL || 'tts-1',
        whisperModel: process.env.WHISPER_MODEL || 'whisper-1',
      },
    });
  });

  // ─── Chat Endpoint (non-streaming) ────────────────────────────
  router.post('/chat', async (req: Request, res: Response) => {
    try {
      const { message, includeVision } = req.body;

      if (!message) {
        return res.status(400).json({ error: 'Message is required' });
      }

      const response = await neuralNetwork.chat(
        message,
        includeVision ? neuralNetwork.getLastVisionAnalysis() : null
      );

      res.json({
        response,
        timestamp: Date.now(),
        visionContext: includeVision ? neuralNetwork.getLastVisionAnalysis() : null,
      });
    } catch (error) {
      console.error('[API] Chat error:', error);
      res.status(500).json({ error: 'Failed to process chat message' });
    }
  });

  // ─── Audio Processing Endpoint ────────────────────────────────
  router.post('/audio/transcribe', upload.single('audio'), async (req: Request, res: Response) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: 'Audio file is required' });
      }

      const transcription = await neuralNetwork.transcribeAudio(
        req.file.buffer,
        req.file.originalname.split('.').pop() || 'webm'
      );

      res.json({
        transcription,
        timestamp: Date.now(),
      });
    } catch (error) {
      console.error('[API] Transcription error:', error);
      res.status(500).json({ error: 'Failed to transcribe audio' });
    }
  });

  // ─── Text-to-Speech Endpoint ──────────────────────────────────
  router.post('/audio/synthesize', async (req: Request, res: Response) => {
    try {
      const { text } = req.body;

      if (!text) {
        return res.status(400).json({ error: 'Text is required' });
      }

      const result = await neuralNetwork.synthesizeSpeech(text);

      res.set({
        'Content-Type': 'audio/mpeg',
        'Content-Length': result.audioBuffer.length.toString(),
      });
      res.send(result.audioBuffer);
    } catch (error) {
      console.error('[API] TTS error:', error);
      res.status(500).json({ error: 'Failed to synthesize speech' });
    }
  });

  // ─── Full Multimodal Interaction ──────────────────────────────
  router.post('/interact', upload.single('audio'), async (req: Request, res: Response) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: 'Audio file is required' });
      }

      const frameBase64 = req.body.frame;
      let frame: FrameData | undefined;

      if (frameBase64) {
        frame = {
          imageBase64: frameBase64,
          timestamp: Date.now(),
          width: 640,
          height: 480,
        };
      }

      const result = await neuralNetwork.processMultimodal(
        req.file.buffer,
        req.file.originalname.split('.').pop() || 'webm',
        frame
      );

      res.json({
        text: result.text,
        audio: result.audioBuffer.toString('base64'),
        audioFormat: 'mp3',
        visionAnalysis: result.visionAnalysis || null,
        timestamp: Date.now(),
      });
    } catch (error) {
      console.error('[API] Interaction error:', error);
      res.status(500).json({ error: 'Failed to process interaction' });
    }
  });

  // ─── Vision Analysis Endpoint ─────────────────────────────────
  router.post('/vision/analyze', async (req: Request, res: Response) => {
    try {
      const { frame } = req.body;

      if (!frame) {
        return res.status(400).json({ error: 'Frame data is required' });
      }

      const frameData: FrameData = {
        imageBase64: frame,
        timestamp: Date.now(),
        width: parseInt(req.body.width) || 640,
        height: parseInt(req.body.height) || 480,
      };

      const analysis = await neuralNetwork.analyzeFrame(frameData);

      res.json({
        analysis,
        timestamp: Date.now(),
      });
    } catch (error) {
      console.error('[API] Vision analysis error:', error);
      res.status(500).json({ error: 'Failed to analyze frame' });
    }
  });

  // ─── Describe Current Scene ───────────────────────────────────
  router.get('/vision/describe', async (_req: Request, res: Response) => {
    try {
      const description = await neuralNetwork.describeActivity();
      res.json({
        description,
        timestamp: Date.now(),
      });
    } catch (error) {
      console.error('[API] Scene description error:', error);
      res.status(500).json({ error: 'Failed to describe scene' });
    }
  });

  // ─── Conversation History ─────────────────────────────────────
  router.get('/conversation', (_req: Request, res: Response) => {
    res.json({
      messages: neuralNetwork.getConversationHistory(),
      timestamp: Date.now(),
    });
  });

  router.delete('/conversation', (_req: Request, res: Response) => {
    neuralNetwork.clearHistory();
    res.json({ success: true, timestamp: Date.now() });
  });

  return router;
}
