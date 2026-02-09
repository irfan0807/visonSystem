/**
 * ChatPanel — Conversation interface with the AI
 * Supports text input and displays streaming responses
 */

import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Sparkles, User, Bot } from 'lucide-react';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  isStreaming?: boolean;
}

interface ChatPanelProps {
  messages: ChatMessage[];
  onSendMessage: (text: string) => void;
  isThinking: boolean;
  disabled: boolean;
}

export const ChatPanel: React.FC<ChatPanelProps> = ({
  messages,
  onSendMessage,
  isThinking,
  disabled,
}) => {
  const [input, setInput] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isThinking]);

  const handleSend = () => {
    const text = input.trim();
    if (!text || disabled) return;
    onSendMessage(text);
    setInput('');
    inputRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-dark-700">
        <Sparkles className="w-4 h-4 text-neural-400" />
        <h2 className="text-sm font-semibold text-dark-200">Conversation</h2>
        <span className="text-[10px] text-dark-500 ml-auto">{messages.length} messages</span>
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center gap-3 opacity-50">
            <Bot className="w-10 h-10 text-neural-500" />
            <p className="text-sm text-dark-400">
              Start talking or typing to interact with IRIS.
              <br />
              <span className="text-xs text-dark-500">
                I can see through your camera and hear your voice.
              </span>
            </p>
          </div>
        )}

        <AnimatePresence initial={false}>
          {messages.map((msg) => (
            <motion.div
              key={msg.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2 }}
              className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
            >
              {/* Avatar */}
              <div
                className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                  msg.role === 'user'
                    ? 'bg-iris-600/20 border border-iris-500/30'
                    : 'bg-neural-600/20 border border-neural-500/30'
                }`}
              >
                {msg.role === 'user' ? (
                  <User className="w-4 h-4 text-iris-400" />
                ) : (
                  <Bot className="w-4 h-4 text-neural-400" />
                )}
              </div>

              {/* Message bubble */}
              <div
                className={`max-w-[80%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
                  msg.role === 'user'
                    ? 'bg-iris-600/20 border border-iris-500/20 text-dark-100'
                    : 'bg-dark-800 border border-dark-700 text-dark-200'
                }`}
              >
                {msg.content}
                {msg.isStreaming && (
                  <motion.span
                    animate={{ opacity: [1, 0, 1] }}
                    transition={{ duration: 0.8, repeat: Infinity }}
                    className="inline-block ml-1 w-1.5 h-4 bg-neural-400 rounded-sm align-middle"
                  />
                )}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Thinking indicator */}
        {isThinking && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex gap-3"
          >
            <div className="w-8 h-8 rounded-full flex items-center justify-center bg-neural-600/20 border border-neural-500/30">
              <Bot className="w-4 h-4 text-neural-400" />
            </div>
            <div className="bg-dark-800 border border-dark-700 rounded-2xl px-4 py-3 flex gap-1.5">
              <div className="typing-dot" />
              <div className="typing-dot" />
              <div className="typing-dot" />
            </div>
          </motion.div>
        )}
      </div>

      {/* Input */}
      <div className="p-3 border-t border-dark-700">
        <div className="flex items-center gap-2 bg-dark-800 rounded-xl border border-dark-700 focus-within:border-neural-500/50 transition-colors px-3">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type a message..."
            disabled={disabled}
            className="flex-1 bg-transparent py-3 text-sm text-dark-100 placeholder-dark-500 focus:outline-none"
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || disabled}
            className="p-2 rounded-lg text-neural-400 hover:text-neural-300 hover:bg-neural-500/10 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};
