/**
 * MaskOverlay Component
 * Renders segmentation masks over the camera view
 */

import React from 'react';
import { View, StyleSheet, Text } from 'react-native';
import Svg, { Polygon, Rect, G, Text as SvgText } from 'react-native-svg';
import type { MaskData, BoundingBox, Point } from '../types';
import { UI_CONFIG } from '../config';

interface MaskOverlayProps {
  masks: MaskData[];
  frameWidth: number;
  frameHeight: number;
  viewWidth: number;
  viewHeight: number;
  showLabels?: boolean;
  showBoundingBoxes?: boolean;
}

export const MaskOverlay: React.FC<MaskOverlayProps> = ({
  masks,
  frameWidth,
  frameHeight,
  viewWidth,
  viewHeight,
  showLabels = true,
  showBoundingBoxes = true,
}) => {
  // Calculate scale factors
  const scaleX = viewWidth / frameWidth;
  const scaleY = viewHeight / frameHeight;

  /**
   * Scale a point from frame coordinates to view coordinates
   */
  const scalePoint = (point: Point): Point => ({
    x: point.x * scaleX,
    y: point.y * scaleY,
  });

  /**
   * Scale a bounding box from frame coordinates to view coordinates
   */
  const scaleBoundingBox = (box: BoundingBox): BoundingBox => ({
    x: box.x * scaleX,
    y: box.y * scaleY,
    width: box.width * scaleX,
    height: box.height * scaleY,
  });

  /**
   * Get mask color based on confidence
   */
  const getMaskColor = (confidence: number): string => {
    if (confidence >= 0.8) return UI_CONFIG.maskColors.high;
    if (confidence >= 0.5) return UI_CONFIG.maskColors.medium;
    return UI_CONFIG.maskColors.low;
  };

  /**
   * Convert polygon points to SVG path string
   */
  const getPolygonPoints = (polygon: Point[]): string => {
    return polygon
      .map(p => {
        const scaled = scalePoint(p);
        return `${scaled.x},${scaled.y}`;
      })
      .join(' ');
  };

  return (
    <View style={[StyleSheet.absoluteFill, styles.container]}>
      <Svg width={viewWidth} height={viewHeight}>
        {masks.map((mask, index) => {
          const color = mask.color || getMaskColor(mask.confidence);
          const scaledBox = scaleBoundingBox(mask.boundingBox);

          return (
            <G key={mask.id || index}>
              {/* Polygon mask */}
              {mask.polygon && mask.polygon.length > 0 && (
                <Polygon
                  points={getPolygonPoints(mask.polygon)}
                  fill={color}
                  stroke={color.replace('0.4', '1')}
                  strokeWidth={2}
                />
              )}

              {/* Bounding box */}
              {showBoundingBoxes && (
                <Rect
                  x={scaledBox.x}
                  y={scaledBox.y}
                  width={scaledBox.width}
                  height={scaledBox.height}
                  fill="none"
                  stroke={color.replace('0.4', '0.8')}
                  strokeWidth={2}
                  strokeDasharray="5,5"
                />
              )}

              {/* Label */}
              {showLabels && (
                <>
                  <Rect
                    x={scaledBox.x}
                    y={scaledBox.y - 24}
                    width={Math.max(mask.label.length * 8 + 40, 80)}
                    height={22}
                    fill="rgba(0, 0, 0, 0.7)"
                    rx={4}
                    ry={4}
                  />
                  <SvgText
                    x={scaledBox.x + 6}
                    y={scaledBox.y - 8}
                    fill="#fff"
                    fontSize={12}
                    fontWeight="bold"
                  >
                    {`${mask.label} ${(mask.confidence * 100).toFixed(0)}%`}
                  </SvgText>
                </>
              )}
            </G>
          );
        })}
      </Svg>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    pointerEvents: 'none',
  },
});

export default MaskOverlay;
