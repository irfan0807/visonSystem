/**
 * VisionPanel — Shows what the AI currently "sees" and detects
 */

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Eye, Box, AlertTriangle, Smile } from 'lucide-react';
import type { VisionAnalysis } from '../hooks/useSocket';

interface VisionPanelProps {
  analysis: VisionAnalysis | null;
  isAnalyzing: boolean;
}

export const VisionPanel: React.FC<VisionPanelProps> = ({ analysis, isAnalyzing }) => {
  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-dark-700">
        <Eye className="w-4 h-4 text-iris-400" />
        <h2 className="text-sm font-semibold text-dark-200">Vision Analysis</h2>
        {isAnalyzing && (
          <motion.div
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 1.5, repeat: Infinity }}
            className="ml-auto flex items-center gap-1"
          >
            <div className="w-1.5 h-1.5 rounded-full bg-iris-400" />
            <span className="text-[10px] text-iris-400">Analyzing</span>
          </motion.div>
        )}
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {!analysis ? (
          <div className="flex flex-col items-center justify-center h-full text-center gap-3 opacity-50">
            <Eye className="w-8 h-8 text-dark-600" />
            <p className="text-xs text-dark-500">
              Enable camera to see vision analysis
            </p>
          </div>
        ) : (
          <AnimatePresence mode="wait">
            <motion.div
              key={analysis.timestamp}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="space-y-4"
            >
              {/* Scene Description */}
              <div className="space-y-1.5">
                <p className="text-[10px] font-semibold text-dark-400 uppercase tracking-wider">
                  Scene
                </p>
                <p className="text-sm text-dark-200 leading-relaxed">
                  {analysis.description}
                </p>
                <span className="inline-block text-[10px] bg-dark-800 text-dark-400 rounded-full px-2 py-0.5">
                  {analysis.scene}
                </span>
              </div>

              {/* Detected Objects */}
              {analysis.objects.length > 0 && (
                <div className="space-y-2">
                  <div className="flex items-center gap-1.5">
                    <Box className="w-3.5 h-3.5 text-dark-400" />
                    <p className="text-[10px] font-semibold text-dark-400 uppercase tracking-wider">
                      Objects ({analysis.objects.length})
                    </p>
                  </div>
                  <div className="space-y-1">
                    {analysis.objects.map((obj, i) => (
                      <div
                        key={`${obj.label}-${i}`}
                        className="flex items-center justify-between bg-dark-800/50 rounded-lg px-3 py-1.5"
                      >
                        <span className="text-xs text-dark-200">{obj.label}</span>
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-1.5 bg-dark-700 rounded-full overflow-hidden">
                            <motion.div
                              initial={{ width: 0 }}
                              animate={{ width: `${obj.confidence * 100}%` }}
                              className={`h-full rounded-full ${
                                obj.confidence > 0.8
                                  ? 'bg-green-500'
                                  : obj.confidence > 0.5
                                    ? 'bg-yellow-500'
                                    : 'bg-red-500'
                              }`}
                            />
                          </div>
                          <span className="text-[10px] text-dark-500 w-8 text-right">
                            {(obj.confidence * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Emotions */}
              {analysis.emotions && analysis.emotions.length > 0 && (
                <div className="space-y-2">
                  <div className="flex items-center gap-1.5">
                    <Smile className="w-3.5 h-3.5 text-dark-400" />
                    <p className="text-[10px] font-semibold text-dark-400 uppercase tracking-wider">
                      Emotions
                    </p>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {analysis.emotions.map((e, i) => (
                      <span
                        key={i}
                        className="text-xs bg-neural-500/10 border border-neural-500/20 text-neural-300 rounded-full px-2.5 py-1"
                      >
                        {e.emotion} {(e.confidence * 100).toFixed(0)}%
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Anomalies */}
              {analysis.anomalies.length > 0 && (
                <div className="space-y-2">
                  <div className="flex items-center gap-1.5">
                    <AlertTriangle className="w-3.5 h-3.5 text-red-400" />
                    <p className="text-[10px] font-semibold text-red-400 uppercase tracking-wider">
                      Anomalies
                    </p>
                  </div>
                  {analysis.anomalies.map((a, i) => (
                    <div
                      key={i}
                      className="text-xs bg-red-500/10 border border-red-500/20 text-red-300 rounded-lg px-3 py-2"
                    >
                      {a}
                    </div>
                  ))}
                </div>
              )}
            </motion.div>
          </AnimatePresence>
        )}
      </div>
    </div>
  );
};
