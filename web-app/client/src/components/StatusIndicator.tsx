/**
 * StatusIndicator — Shows connection and system status
 */

import React from 'react';
import { motion } from 'framer-motion';
import { Wifi, WifiOff, Brain, Eye, Ear } from 'lucide-react';

interface StatusIndicatorProps {
  connected: boolean;
  cameraActive: boolean;
  micActive: boolean;
  aiThinking: boolean;
}

export const StatusIndicator: React.FC<StatusIndicatorProps> = ({
  connected,
  cameraActive,
  micActive,
  aiThinking,
}) => {
  return (
    <div className="flex items-center gap-3">
      {/* Connection */}
      <div className="flex items-center gap-1.5" title={connected ? 'Connected to server' : 'Disconnected'}>
        {connected ? (
          <Wifi className="w-3.5 h-3.5 text-green-400" />
        ) : (
          <WifiOff className="w-3.5 h-3.5 text-red-400" />
        )}
        <span className={`text-[10px] font-medium ${connected ? 'text-green-400' : 'text-red-400'}`}>
          {connected ? 'ONLINE' : 'OFFLINE'}
        </span>
      </div>

      <div className="w-px h-4 bg-dark-700" />

      {/* Camera */}
      <div className="flex items-center gap-1" title={cameraActive ? 'Camera active' : 'Camera off'}>
        <Eye className={`w-3.5 h-3.5 ${cameraActive ? 'text-iris-400' : 'text-dark-600'}`} />
      </div>

      {/* Microphone */}
      <div className="flex items-center gap-1" title={micActive ? 'Microphone active' : 'Microphone off'}>
        <Ear className={`w-3.5 h-3.5 ${micActive ? 'text-iris-400' : 'text-dark-600'}`} />
      </div>

      {/* Neural Network */}
      <div className="flex items-center gap-1" title="Neural network">
        {aiThinking ? (
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
          >
            <Brain className="w-3.5 h-3.5 text-neural-400" />
          </motion.div>
        ) : (
          <Brain className="w-3.5 h-3.5 text-dark-600" />
        )}
      </div>
    </div>
  );
};
