/**
 * TypeScript types matching backend Pydantic schemas
 */

export type ObjectType = 'movable' | 'structural';

export interface RoomObject {
  id: string;
  label: string;
  bbox: [number, number, number, number]; // [x, y, width, height] in pixels
  type: ObjectType;
  orientation: number;
  is_locked: boolean;
}

export interface RoomDimensions {
  width_estimate: number;
  height_estimate: number;
}

export interface ConstraintViolation {
  constraint_name: string;
  description: string;
  severity: 'error' | 'warning';
  objects_involved: string[];
}

// === API Request/Response Types ===

export interface AnalyzeRequest {
  image_base64: string;
}

export interface AnalyzeResponse {
  room_dimensions: RoomDimensions;
  objects: RoomObject[];
  detected_issues: string[];
  message: string;
}

export interface OptimizeRequest {
  current_layout: RoomObject[];
  locked_ids: string[];
  room_dimensions: RoomDimensions;
  max_iterations?: number;
}

export interface OptimizeResponse {
  new_layout: RoomObject[];
  explanation: string;
  layout_score: number;
  iterations: number;
  constraint_violations: ConstraintViolation[];
  improvement: number;
}

export interface RenderRequest {
  original_image_base64: string;
  final_layout: RoomObject[];
  original_layout: RoomObject[];
}

export interface RenderResponse {
  image_url: string | null;
  image_base64: string | null;
  message: string;
}

// === Frontend State ===

export interface ClearanceZone {
  object_id: string;
  bounds: [number, number, number, number];
  type: string;
}

export interface Overlays {
  walking_paths?: number[][];
  clearance_zones?: ClearanceZone[];
}

export interface AppState {
  // Image
  image: string | null;
  imageId: string | null;
  renderedImage: string | null; // After optimization render
  showComparison: boolean; // Show before/after view

  // Room data
  roomDimensions: RoomDimensions | null;
  objects: RoomObject[];
  originalObjects: RoomObject[]; // For comparison

  // Selection
  selectedObjectId: string | null;
  lockedObjectIds: string[];

  // Loading states
  isAnalyzing: boolean;
  isOptimizing: boolean;
  isRendering: boolean;

  // Results
  explanation: string;
  violations: string[];
  layoutScore: number | null;

  // Overlays
  overlays: Overlays;

  // Edit mode
  maskMode: boolean;
  editMasks: EditMask[];
}

export interface EditMask {
  region_mask: string; // base64 PNG
  instruction: string;
}

// === Object Icons ===
export const OBJECT_ICONS: Record<string, string> = {
  bed: '🛏️',
  desk: '🪑',
  chair: '💺',
  door: '🚪',
  window: '🪟',
  wardrobe: '🚪',
  nightstand: '🛋️',
  dresser: '🗄️',
  default: '📦'
};

export function getObjectIcon(label: string): string {
  return OBJECT_ICONS[label.toLowerCase()] || OBJECT_ICONS.default;
}
