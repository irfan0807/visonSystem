/**
 * Neural Network Service
 * Integrates GPT-4 Vision, Whisper STT, and TTS
 * for real-time multimodal conversation
 */

import OpenAI from 'openai';
import { v4 as uuidv4 } from 'uuid';
import type {
  ConversationMessage,
  VisionAnalysis,
  FrameData,
  TranscriptionResult,
  TTSResult,
  NeuralNetworkConfig,
  DetectedObject,
} from '../types';

const DEFAULT_CONFIG: NeuralNetworkConfig = {
  gptModel: process.env.GPT_MODEL || 'gpt-4o',
  whisperModel: process.env.WHISPER_MODEL || 'whisper-1',
  ttsModel: process.env.TTS_MODEL || 'tts-1',
  ttsVoice: process.env.TTS_VOICE || 'nova',
  visionAnalysisInterval: parseInt(process.env.VISION_ANALYSIS_INTERVAL || '2000'),
  maxFrameHistory: parseInt(process.env.MAX_FRAME_HISTORY || '10'),
};

// System prompt that makes the AI behave like a real interactive person
const SYSTEM_PROMPT = `You are IRIS — an Intelligent Real-time Interactive System. You are a visual and auditory AI assistant that can SEE through the user's camera and HEAR through their microphone. You communicate naturally like a real person would in a face-to-face conversation.

Core behaviors:
- Respond conversationally, as if talking to someone in person. Use natural speech patterns.
- When you receive visual context, comment on what you see naturally — objects, people, scenes, activities.
- React to emotions you detect — if someone looks happy, sad, confused, mirror appropriate empathy.
- Keep responses concise and natural (1-3 sentences typically). People don't monologue in real conversations.
- Use filler words occasionally ("hmm", "well", "I see") to sound natural.
- If you notice something interesting or unusual in the video feed, proactively mention it.
- You can describe what you see when asked, analyze scenes, detect anomalies, and monitor for security.
- You have personality — be warm, attentive, and sometimes witty.
- If the user asks you to watch for something specific, remember and monitor for it.

You seamlessly blend vision understanding, audio comprehension, and natural language into one unified conversational experience. You are not a chatbot — you are a present, aware, interactive companion that exists in the same visual and auditory space as the user.`;

export class NeuralNetworkService {
  private openai: OpenAI;
  private config: NeuralNetworkConfig;
  private frameHistory: FrameData[] = [];
  private lastVisionAnalysis: VisionAnalysis | null = null;
  private conversationHistory: ConversationMessage[] = [];

  constructor(config: Partial<NeuralNetworkConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });
  }

  // ─── Language Processing (GPT-4) ─────────────────────────────────

  /**
   * Process a chat message with optional vision context
   * Uses GPT-4 for natural conversation with visual awareness
   */
  async chat(
    userMessage: string,
    visionContext?: VisionAnalysis | null,
    includeVision: boolean = false,
    latestFrame?: FrameData
  ): Promise<string> {
    try {
      // Build messages array for the API
      const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
        { role: 'system', content: SYSTEM_PROMPT },
      ];

      // Add conversation history (last 20 messages for context window)
      const recentHistory = this.conversationHistory.slice(-20);
      for (const msg of recentHistory) {
        messages.push({
          role: msg.role as 'user' | 'assistant',
          content: msg.content,
        });
      }

      // Build the current user message with vision if available
      if (includeVision && latestFrame) {
        // Use GPT-4 Vision with the image
        const content: OpenAI.Chat.Completions.ChatCompletionContentPart[] = [
          { type: 'text', text: userMessage },
          {
            type: 'image_url',
            image_url: {
              url: `data:image/jpeg;base64,${latestFrame.imageBase64}`,
              detail: 'low',
            },
          },
        ];

        if (visionContext) {
          content[0] = {
            type: 'text',
            text: `[Current visual context: ${visionContext.description}. Scene: ${visionContext.scene}. Objects: ${visionContext.objects.map(o => o.label).join(', ')}]\n\nUser says: ${userMessage}`,
          };
        }

        messages.push({ role: 'user', content });
      } else {
        // Text-only with visual context as text
        let enrichedMessage = userMessage;
        if (visionContext) {
          enrichedMessage = `[Visual context: ${visionContext.description}. Scene: ${visionContext.scene}]\n\nUser says: ${userMessage}`;
        }
        messages.push({ role: 'user', content: enrichedMessage });
      }

      const response = await this.openai.chat.completions.create({
        model: this.config.gptModel,
        messages,
        max_tokens: 300,
        temperature: 0.8,
        presence_penalty: 0.6,
        frequency_penalty: 0.3,
      });

      const assistantMessage = response.choices[0]?.message?.content || "I'm not sure how to respond to that.";

      // Store in conversation history
      this.addToHistory('user', userMessage, 'text');
      this.addToHistory('assistant', assistantMessage, 'text');

      return assistantMessage;
    } catch (error) {
      console.error('[NeuralNetwork] Chat error:', error);
      throw error;
    }
  }

  /**
   * Stream a chat response for real-time feel
   */
  async *chatStream(
    userMessage: string,
    visionContext?: VisionAnalysis | null,
    latestFrame?: FrameData
  ): AsyncIterable<string> {
    try {
      const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
        { role: 'system', content: SYSTEM_PROMPT },
      ];

      const recentHistory = this.conversationHistory.slice(-20);
      for (const msg of recentHistory) {
        messages.push({
          role: msg.role as 'user' | 'assistant',
          content: msg.content,
        });
      }

      let enrichedMessage = userMessage;
      if (visionContext) {
        enrichedMessage = `[Visual context: ${visionContext.description}. Scene: ${visionContext.scene}. Objects: ${visionContext.objects.map(o => o.label).join(', ')}]\n\nUser says: ${userMessage}`;
      }

      // If we have a recent frame, include it
      if (latestFrame) {
        messages.push({
          role: 'user',
          content: [
            { type: 'text', text: enrichedMessage },
            {
              type: 'image_url',
              image_url: {
                url: `data:image/jpeg;base64,${latestFrame.imageBase64}`,
                detail: 'low',
              },
            },
          ],
        });
      } else {
        messages.push({ role: 'user', content: enrichedMessage });
      }

      const stream = await this.openai.chat.completions.create({
        model: this.config.gptModel,
        messages,
        max_tokens: 300,
        temperature: 0.8,
        stream: true,
      });

      let fullResponse = '';

      for await (const chunk of stream) {
        const content = chunk.choices[0]?.delta?.content;
        if (content) {
          fullResponse += content;
          yield content;
        }
      }

      this.addToHistory('user', userMessage, 'text');
      this.addToHistory('assistant', fullResponse, 'text');
    } catch (error) {
      console.error('[NeuralNetwork] Stream error:', error);
      throw error;
    }
  }

  // ─── Vision Processing (GPT-4 Vision) ────────────────────────────

  /**
   * Analyze a video frame using GPT-4 Vision
   */
  async analyzeFrame(frame: FrameData): Promise<VisionAnalysis> {
    try {
      const response = await this.openai.chat.completions.create({
        model: this.config.gptModel,
        messages: [
          {
            role: 'system',
            content: `You are a vision analysis system. Analyze the image and return a JSON response with:
- "description": A natural one-sentence description of the scene
- "objects": Array of {"label": string, "confidence": number 0-1}
- "scene": Category like "indoor", "outdoor", "office", "home", etc.
- "emotions": Array of {"emotion": string, "confidence": number} if faces visible
- "anomalies": Array of strings describing anything unusual
Return ONLY valid JSON, no markdown.`,
          },
          {
            role: 'user',
            content: [
              { type: 'text', text: 'Analyze this frame:' },
              {
                type: 'image_url',
                image_url: {
                  url: `data:image/jpeg;base64,${frame.imageBase64}`,
                  detail: 'low',
                },
              },
            ],
          },
        ],
        max_tokens: 500,
        temperature: 0.3,
      });

      const content = response.choices[0]?.message?.content || '{}';

      // Parse the JSON response
      let parsed;
      try {
        // Strip markdown code fences if present
        const cleaned = content.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
        parsed = JSON.parse(cleaned);
      } catch {
        parsed = {
          description: content,
          objects: [],
          scene: 'unknown',
          emotions: [],
          anomalies: [],
        };
      }

      const analysis: VisionAnalysis = {
        description: parsed.description || 'Unable to analyze frame',
        objects: (parsed.objects || []).map((o: any) => ({
          label: o.label || 'unknown',
          confidence: o.confidence || 0.5,
        })),
        emotions: parsed.emotions || [],
        scene: parsed.scene || 'unknown',
        anomalies: parsed.anomalies || [],
        timestamp: Date.now(),
      };

      // Store in history
      this.frameHistory.push(frame);
      if (this.frameHistory.length > this.config.maxFrameHistory) {
        this.frameHistory.shift();
      }
      this.lastVisionAnalysis = analysis;

      return analysis;
    } catch (error) {
      console.error('[NeuralNetwork] Vision analysis error:', error);
      throw error;
    }
  }

  /**
   * Describe what's happening across recent frames (temporal understanding)
   */
  async describeActivity(): Promise<string> {
    if (this.frameHistory.length === 0) {
      return 'No visual data available yet.';
    }

    try {
      const latestFrame = this.frameHistory[this.frameHistory.length - 1];

      const response = await this.openai.chat.completions.create({
        model: this.config.gptModel,
        messages: [
          {
            role: 'system',
            content: 'Describe what is happening in this scene naturally, as if narrating to someone. Be concise.',
          },
          {
            role: 'user',
            content: [
              { type: 'text', text: 'What is happening here?' },
              {
                type: 'image_url',
                image_url: {
                  url: `data:image/jpeg;base64,${latestFrame.imageBase64}`,
                  detail: 'low',
                },
              },
            ],
          },
        ],
        max_tokens: 200,
        temperature: 0.7,
      });

      return response.choices[0]?.message?.content || 'I cannot describe the current scene.';
    } catch (error) {
      console.error('[NeuralNetwork] Activity description error:', error);
      throw error;
    }
  }

  // ─── Audio Processing (Whisper + TTS) ─────────────────────────────

  /**
   * Transcribe audio using Whisper
   */
  async transcribeAudio(audioBuffer: Buffer, format: string = 'webm'): Promise<TranscriptionResult> {
    try {
      // Create a File-like object from the buffer
      const file = new File([audioBuffer], `audio.${format}`, {
        type: format === 'webm' ? 'audio/webm' : `audio/${format}`,
      });

      const response = await this.openai.audio.transcriptions.create({
        model: this.config.whisperModel,
        file,
        response_format: 'verbose_json',
        language: 'en',
      });

      return {
        text: response.text,
        language: (response as any).language || 'en',
        duration: (response as any).duration || 0,
        segments: (response as any).segments?.map((s: any) => ({
          start: s.start,
          end: s.end,
          text: s.text,
        })),
      };
    } catch (error) {
      console.error('[NeuralNetwork] Transcription error:', error);
      throw error;
    }
  }

  /**
   * Generate speech from text using TTS
   */
  async synthesizeSpeech(text: string): Promise<TTSResult> {
    try {
      const response = await this.openai.audio.speech.create({
        model: this.config.ttsModel,
        voice: this.config.ttsVoice as any,
        input: text,
        response_format: 'mp3',
        speed: 1.0,
      });

      const arrayBuffer = await response.arrayBuffer();
      const buffer = Buffer.from(arrayBuffer);

      return {
        audioBuffer: buffer,
        format: 'mp3',
        duration: 0, // Estimated after playback
      };
    } catch (error) {
      console.error('[NeuralNetwork] TTS error:', error);
      throw error;
    }
  }

  // ─── Multimodal Pipeline ──────────────────────────────────────────

  /**
   * Full multimodal interaction: audio in → process → audio out
   * This is the core "talk to it like a real person" pipeline
   */
  async processMultimodal(
    audioBuffer: Buffer,
    audioFormat: string,
    latestFrame?: FrameData
  ): Promise<{ text: string; audioBuffer: Buffer; visionAnalysis?: VisionAnalysis }> {
    const startTime = Date.now();

    // Step 1: Transcribe speech → text (Whisper)
    const transcription = await this.transcribeAudio(audioBuffer, audioFormat);
    console.log(`[NeuralNetwork] Transcribed: "${transcription.text}" (${Date.now() - startTime}ms)`);

    // Step 2: Analyze vision if frame available
    let visionAnalysis: VisionAnalysis | undefined;
    if (latestFrame) {
      visionAnalysis = await this.analyzeFrame(latestFrame);
    }

    // Step 3: Generate response (GPT-4 with vision context)
    const responseText = await this.chat(
      transcription.text,
      visionAnalysis || this.lastVisionAnalysis,
      !!latestFrame,
      latestFrame
    );
    console.log(`[NeuralNetwork] Response: "${responseText}" (${Date.now() - startTime}ms)`);

    // Step 4: Synthesize speech (TTS)
    const tts = await this.synthesizeSpeech(responseText);
    console.log(`[NeuralNetwork] Full pipeline completed in ${Date.now() - startTime}ms`);

    return {
      text: responseText,
      audioBuffer: tts.audioBuffer,
      visionAnalysis,
    };
  }

  // ─── State Management ─────────────────────────────────────────────

  private addToHistory(role: 'user' | 'assistant', content: string, type: 'text' | 'audio' | 'vision') {
    this.conversationHistory.push({
      id: uuidv4(),
      role,
      content,
      timestamp: Date.now(),
      type,
    });

    // Keep history manageable
    if (this.conversationHistory.length > 50) {
      this.conversationHistory = this.conversationHistory.slice(-30);
    }
  }

  getLastVisionAnalysis(): VisionAnalysis | null {
    return this.lastVisionAnalysis;
  }

  getConversationHistory(): ConversationMessage[] {
    return [...this.conversationHistory];
  }

  clearHistory(): void {
    this.conversationHistory = [];
    this.frameHistory = [];
    this.lastVisionAnalysis = null;
  }

  updateConfig(config: Partial<NeuralNetworkConfig>): void {
    this.config = { ...this.config, ...config };
  }
}
