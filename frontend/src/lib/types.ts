/**
 * TypeScript types matching backend Pydantic schemas
 * 
 * Simplified for Generative Interior Design Agent
 */

export type ObjectType = 'movable' | 'structural';

export interface RoomObject {
  id: string;
  label: string;
  bbox: [number, number, number, number]; // [x, y, width, height] in pixels/percentage
  type: ObjectType;
  orientation: number; // 0=North, 90=East, 180=South, 270=West
  is_locked: boolean;
  // Fields for 3D understanding
  z_index?: number; // 0=floor, 1=furniture, 2=ceiling
  material_hint?: string | null; // wooden, fabric, metal, glass
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
  wall_bounds: [number, number, number, number] | null;
  message: string;
}

export interface OptimizeRequest {
  current_layout: RoomObject[];
  locked_ids: string[];
  room_dimensions: RoomDimensions;
  max_iterations?: number;
  image_base64?: string;
}

// Layout variation from AI Designer
export interface LayoutVariation {
  name: string; // "Productivity Focus", "Cozy Retreat", "Space Optimized"
  description: string; // Design rationale
  layout: RoomObject[];
  layout_plan?: Record<string, any> | null; // Semantic placement plan from Gemini
  thumbnail_base64?: string | null;
  door_info?: Record<string, any> | null;
  window_info?: Record<string, any> | null;
}

export interface OptimizeResponse {
  variations: LayoutVariation[]; // 2-3 layout options
  message: string;
  new_layout?: RoomObject[] | null;
  explanation?: string | null;
  iterations?: number | null;
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
  sofa: '🛋️',
  table: '🪑',
  rug: '🟫',
  lamp: '💡',
  closet: '🚪',
  default: '📦'
};

export function getObjectIcon(label: string): string {
  return OBJECT_ICONS[label.toLowerCase()] || OBJECT_ICONS.default;
}

// === App Stage ===
export type AppStage = 'analyze' | 'layouts' | 'perspective' | 'chat' | 'shop';

// === Chat Edit Types ===
export interface ChatEditRequest {
  command: string;
  current_layout: RoomObject[];
  room_dimensions: RoomDimensions;
  current_image_base64?: string;
  layout_plan?: Record<string, any> | null;
}

export interface ChatEditResponse {
  edit_type: 'layout' | 'cosmetic' | 'replace' | 'remove';
  updated_layout: RoomObject[];
  updated_image_base64: string | null;
  explanation: string;
  needs_rerender: boolean;
}

// === Perspective Types ===
export interface PerspectiveRequest {
  layout: RoomObject[];
  room_dimensions: RoomDimensions;
  style?: string;
  view_angle?: string;
  image_base64?: string;
  layout_plan?: Record<string, any> | null;
  door_info?: Record<string, any> | null;
  window_info?: Record<string, any> | null;
}

export interface PerspectiveResponse {
  image_base64: string | null;
  message: string;
}

export interface ShopRequest {
  current_layout: RoomObject[];
  total_budget: number;
  perspective_image_base64?: string;
}

export interface ShopProduct {
  title: string;
  price: number | null;
  price_raw: string;
  link: string;
  thumbnail: string;
  source: string;
  rating: number | null;
  reviews: number | null;
}

export interface ShopItemResult {
  furniture_id: string;
  furniture_label: string;
  search_query: string;
  budget_allocated: number;
  products: ShopProduct[];
  error: string | null;
}

export interface ShopResponse {
  items: ShopItemResult[];
  total_estimated: number;
  total_budget: number;
  message: string;
}

