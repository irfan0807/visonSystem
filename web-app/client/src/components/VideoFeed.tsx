/**
 * VideoFeed — Webcam display with vision analysis overlay
 */

import React, { useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Camera, CameraOff, Eye, EyeOff } from 'lucide-react';
import type { VisionAnalysis } from '../hooks/useSocket';

interface VideoFeedProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  isActive: boolean;
  visionAnalysis: VisionAnalysis | null;
  showOverlay: boolean;
  onToggleCamera: () => void;
  onToggleOverlay: () => void;
}

export const VideoFeed: React.FC<VideoFeedProps> = ({
  videoRef,
  canvasRef,
  isActive,
  visionAnalysis,
  showOverlay,
  onToggleCamera,
  onToggleOverlay,
}) => {
  return (
    <div className="relative w-full h-full rounded-2xl overflow-hidden bg-dark-900 border border-dark-700">
      {/* Video Element */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className={`w-full h-full object-cover ${isActive ? '' : 'hidden'}`}
        style={{ transform: 'scaleX(-1)' }}
      />

      {/* Hidden canvas for frame capture */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Camera Off State */}
      {!isActive && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-4">
          <CameraOff className="w-12 h-12 text-dark-500" />
          <p className="text-dark-400 text-sm">Camera is off</p>
          <button
            onClick={onToggleCamera}
            className="px-4 py-2 bg-neural-600 hover:bg-neural-500 rounded-lg text-sm font-medium transition-colors"
          >
            Enable Camera
          </button>
        </div>
      )}

      {/* Vision Analysis Overlay */}
      <AnimatePresence>
        {isActive && showOverlay && visionAnalysis && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 pointer-events-none"
          >
            {/* Scene description */}
            <div className="absolute top-3 left-3 right-3">
              <div className="glass rounded-lg px-3 py-2">
                <p className="text-xs text-iris-300 font-mono">
                  👁 {visionAnalysis.description}
                </p>
              </div>
            </div>

            {/* Detected objects */}
            <div className="absolute bottom-3 left-3 right-3 flex flex-wrap gap-1.5">
              {visionAnalysis.objects.slice(0, 8).map((obj, i) => (
                <motion.span
                  key={`${obj.label}-${i}`}
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  className="glass rounded-full px-2.5 py-1 text-[10px] font-medium"
                  style={{
                    borderColor:
                      obj.confidence > 0.8
                        ? 'rgba(16, 185, 129, 0.4)'
                        : obj.confidence > 0.5
                          ? 'rgba(245, 158, 11, 0.4)'
                          : 'rgba(239, 68, 68, 0.4)',
                  }}
                >
                  {obj.label}{' '}
                  <span className="text-dark-400">{(obj.confidence * 100).toFixed(0)}%</span>
                </motion.span>
              ))}
            </div>

            {/* Emotion overlay */}
            {visionAnalysis.emotions && visionAnalysis.emotions.length > 0 && (
              <div className="absolute top-3 right-3">
                <div className="glass rounded-lg px-3 py-2">
                  <p className="text-[10px] text-dark-400 mb-1">EMOTION</p>
                  {visionAnalysis.emotions.slice(0, 2).map((e, i) => (
                    <p key={i} className="text-xs font-medium text-neural-300">
                      {e.emotion} {(e.confidence * 100).toFixed(0)}%
                    </p>
                  ))}
                </div>
              </div>
            )}

            {/* Anomaly indicator */}
            {visionAnalysis.anomalies.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="absolute top-14 left-3 right-3"
              >
                <div className="bg-red-500/20 border border-red-500/40 rounded-lg px-3 py-2">
                  <p className="text-xs text-red-300 font-medium">
                    ⚠️ {visionAnalysis.anomalies[0]}
                  </p>
                </div>
              </motion.div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Controls */}
      {isActive && (
        <div className="absolute top-3 right-3 flex gap-2">
          <button
            onClick={onToggleOverlay}
            className="glass rounded-lg p-2 hover:bg-dark-700/60 transition-colors pointer-events-auto"
            title={showOverlay ? 'Hide vision overlay' : 'Show vision overlay'}
          >
            {showOverlay ? (
              <Eye className="w-4 h-4 text-neural-400" />
            ) : (
              <EyeOff className="w-4 h-4 text-dark-400" />
            )}
          </button>
          <button
            onClick={onToggleCamera}
            className="glass rounded-lg p-2 hover:bg-dark-700/60 transition-colors pointer-events-auto"
            title="Toggle camera"
          >
            <CameraOff className="w-4 h-4 text-red-400" />
          </button>
        </div>
      )}

      {/* Live indicator */}
      {isActive && (
        <div className="absolute top-3 left-3 flex items-center gap-2">
          <div className="flex items-center gap-1.5 glass rounded-full px-3 py-1">
            <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
            <span className="text-[10px] font-semibold tracking-wider text-red-400">LIVE</span>
          </div>
        </div>
      )}
    </div>
  );
};
