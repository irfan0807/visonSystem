/**
 * IrisAvatar — The animated AI presence indicator
 * Visual representation of IRIS that reacts to speaking/listening state
 */

import React from 'react';
import { motion } from 'framer-motion';

interface IrisAvatarProps {
  state: 'idle' | 'listening' | 'thinking' | 'speaking';
  size?: 'sm' | 'md' | 'lg';
  audioLevel?: number;
}

const sizeMap = { sm: 48, md: 80, lg: 140 };

const stateColors = {
  idle: { inner: '#8b5cf6', outer: 'rgba(139, 92, 246, 0.2)', ring: 'rgba(139, 92, 246, 0.3)' },
  listening: { inner: '#3aa0ff', outer: 'rgba(58, 160, 255, 0.3)', ring: 'rgba(58, 160, 255, 0.5)' },
  thinking: { inner: '#f59e0b', outer: 'rgba(245, 158, 11, 0.2)', ring: 'rgba(245, 158, 11, 0.3)' },
  speaking: { inner: '#10b981', outer: 'rgba(16, 185, 129, 0.3)', ring: 'rgba(16, 185, 129, 0.5)' },
};

export const IrisAvatar: React.FC<IrisAvatarProps> = ({ state, size = 'md', audioLevel = 0 }) => {
  const px = sizeMap[size];
  const colors = stateColors[state];
  const scale = 1 + audioLevel * 0.3;

  return (
    <div className="relative flex items-center justify-center" style={{ width: px, height: px }}>
      {/* Outer pulsing rings */}
      <motion.div
        className="absolute rounded-full"
        style={{
          width: px * 1.4,
          height: px * 1.4,
          border: `2px solid ${colors.ring}`,
        }}
        animate={{
          scale: state === 'idle' ? [1, 1.1, 1] : [1, 1.15 + audioLevel * 0.2, 1],
          opacity: [0.3, 0.7, 0.3],
        }}
        transition={{ duration: state === 'speaking' ? 0.8 : 2, repeat: Infinity, ease: 'easeInOut' }}
      />
      <motion.div
        className="absolute rounded-full"
        style={{
          width: px * 1.2,
          height: px * 1.2,
          border: `1px solid ${colors.ring}`,
        }}
        animate={{
          scale: [1, 1.08, 1],
          opacity: [0.5, 0.9, 0.5],
        }}
        transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut', delay: 0.3 }}
      />

      {/* Outer glow */}
      <motion.div
        className="absolute rounded-full"
        style={{
          width: px,
          height: px,
          background: `radial-gradient(circle, ${colors.outer} 0%, transparent 70%)`,
        }}
        animate={{ scale: [1, scale, 1] }}
        transition={{ duration: state === 'speaking' ? 0.4 : 1.5, repeat: Infinity }}
      />

      {/* Main orb */}
      <motion.div
        className="absolute rounded-full"
        style={{
          width: px * 0.6,
          height: px * 0.6,
          background: `radial-gradient(circle at 35% 35%, ${colors.inner}, ${colors.inner}aa)`,
          boxShadow: `0 0 ${px * 0.3}px ${colors.inner}66`,
        }}
        animate={{
          scale: state === 'thinking' ? [1, 0.9, 1.1, 1] : [1, scale, 1],
        }}
        transition={{
          duration: state === 'thinking' ? 1 : state === 'speaking' ? 0.5 : 2,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
      />

      {/* Inner eye/core */}
      <motion.div
        className="absolute rounded-full bg-white"
        style={{
          width: px * 0.2,
          height: px * 0.2,
        }}
        animate={{
          scale: state === 'listening' ? [1, 1.2, 1] : 1,
          opacity: state === 'thinking' ? [0.8, 0.4, 0.8] : 0.9,
        }}
        transition={{ duration: 1, repeat: Infinity }}
      />

      {/* State label */}
      {size === 'lg' && (
        <motion.span
          className="absolute -bottom-6 text-xs font-medium tracking-wider uppercase"
          style={{ color: colors.inner }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          key={state}
        >
          {state}
        </motion.span>
      )}
    </div>
  );
};
