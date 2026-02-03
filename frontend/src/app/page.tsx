'use client';

import { useState, useCallback } from 'react';
import { Toaster, toast } from 'react-hot-toast';
import { Moon, Sun, Home } from 'lucide-react';
import { ImageUpload } from '@/components/ImageUpload';
import { CanvasOverlay } from '@/components/CanvasOverlay';
import { ControlPanel } from '@/components/ControlPanel';
import { useAnalyze } from '@/hooks/useAnalyze';
import { useOptimize } from '@/hooks/useOptimize';
import { useRender } from '@/hooks/useRender';
import type { AppState, RoomObject, EditMask } from '@/lib/types';

const initialState: AppState = {
  image: null,
  imageId: null,
  renderedImage: null,
  showComparison: false,
  roomDimensions: null,
  objects: [],
  originalObjects: [],
  selectedObjectId: null,
  lockedObjectIds: [],
  isAnalyzing: false,
  isOptimizing: false,
  isRendering: false,
  explanation: '',
  violations: [],
  layoutScore: null,
  overlays: {},
  maskMode: false,
  editMasks: [],
};

export default function PocketPlannerApp() {
  const [state, setState] = useState<AppState>(initialState);
  const [darkMode, setDarkMode] = useState(true);

  const { analyze } = useAnalyze();
  const { optimize } = useOptimize();
  const { render } = useRender();

  // === Image Upload ===
  const handleImageSelect = useCallback((base64: string) => {
    setState(prev => ({
      ...prev,
      image: `data:image/jpeg;base64,${base64}`,
      imageId: null,
      objects: [],
      originalObjects: [],
      selectedObjectId: null,
      lockedObjectIds: [],
      explanation: '',
      violations: [],
      layoutScore: null,
      overlays: {},
    }));
  }, []);

  // === Analyze ===
  const handleAnalyze = useCallback(async () => {
    if (!state.image) return;

    setState(prev => ({ ...prev, isAnalyzing: true }));

    try {
      const base64 = state.image.split(',')[1];
      const response = await analyze(base64);

      setState(prev => ({
        ...prev,
        roomDimensions: response.room_dimensions,
        objects: response.objects,
        originalObjects: response.objects.map(o => ({ ...o })),
        violations: response.detected_issues,
        isAnalyzing: false,
      }));

      toast.success(`Detected ${response.objects.length} objects`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Analysis failed');
      setState(prev => ({ ...prev, isAnalyzing: false }));
    }
  }, [state.image, analyze]);

  // === Object Selection ===
  const handleObjectSelect = useCallback((id: string) => {
    setState(prev => ({
      ...prev,
      selectedObjectId: prev.selectedObjectId === id ? null : id,
    }));
  }, []);

  // === Object Locking ===
  const handleObjectLock = useCallback((id: string) => {
    setState(prev => {
      const isLocked = prev.lockedObjectIds.includes(id);
      const newLockedIds = isLocked
        ? prev.lockedObjectIds.filter(lockedId => lockedId !== id)
        : [...prev.lockedObjectIds, id];

      return {
        ...prev,
        lockedObjectIds: newLockedIds,
        objects: prev.objects.map(obj => ({
          ...obj,
          is_locked: newLockedIds.includes(obj.id),
        })),
      };
    });
  }, []);

  // === Optimize ===
  const handleOptimize = useCallback(async () => {
    if (!state.roomDimensions || state.lockedObjectIds.length === 0) return;

    setState(prev => ({ ...prev, isOptimizing: true }));

    try {
      const response = await optimize({
        current_layout: state.objects,
        locked_ids: state.lockedObjectIds,
        room_dimensions: state.roomDimensions,
        max_iterations: 5,
      });

      setState(prev => ({
        ...prev,
        objects: response.new_layout,
        explanation: response.explanation,
        layoutScore: response.layout_score,
        violations: response.constraint_violations.map(v => v.description),
        isOptimizing: false,
      }));

      toast.success(`Optimization complete! Score: ${response.layout_score.toFixed(1)}`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Optimization failed');
      setState(prev => ({ ...prev, isOptimizing: false }));
    }
  }, [state.objects, state.roomDimensions, state.lockedObjectIds, optimize]);

  // === Mask Mode ===
  const handleMaskModeToggle = useCallback(() => {
    setState(prev => ({ ...prev, maskMode: !prev.maskMode }));
  }, []);

  // === Edit Mask ===
  const handleAddEdit = useCallback((instruction: string, mask: string) => {
    setState(prev => ({
      ...prev,
      editMasks: [...prev.editMasks, { instruction, region_mask: mask }],
      maskMode: false,
    }));
    toast.success('Edit added');
  }, []);

  const handleRemoveEdit = useCallback((index: number) => {
    setState(prev => ({
      ...prev,
      editMasks: prev.editMasks.filter((_, i) => i !== index),
    }));
  }, []);

  // === Apply Edits ===
  const handleApplyEdits = useCallback(async () => {
    if (!state.image) return;

    setState(prev => ({ ...prev, isRendering: true }));

    try {
      const base64 = state.image.split(',')[1];
      const response = await render({
        original_image_base64: base64,
        final_layout: state.objects,
        original_layout: state.originalObjects,
      });

      if (response.image_base64) {
        setState(prev => ({
          ...prev,
          renderedImage: `data:image/png;base64,${response.image_base64}`,
          showComparison: true,
          editMasks: [],
          isRendering: false,
        }));
        toast.success('Render complete! Showing before/after comparison.');
      } else {
        toast(response.message);
        setState(prev => ({ ...prev, isRendering: false }));
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Rendering failed');
      setState(prev => ({ ...prev, isRendering: false }));
    }
  }, [state.image, state.objects, state.originalObjects, render]);

  // === Mask Complete Handler ===
  const handleMaskComplete = useCallback((maskBase64: string) => {
    const instruction = prompt('Enter edit instruction (e.g., "Repaint wall navy blue"):');
    if (instruction) {
      handleAddEdit(instruction, maskBase64);
    }
  }, [handleAddEdit]);

  return (
    <div className={`min-h-screen ${darkMode ? 'dark bg-slate-950 text-white' : 'bg-gray-50 text-gray-900'}`}>
      <Toaster
        position="top-right"
        toastOptions={{
          className: darkMode ? '!bg-slate-800 !text-white' : '',
        }}
      />

      {/* === Header === */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
              <Home className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                Pocket Planner
              </h1>
              <p className="text-xs text-slate-500">Spatial Optimization for Small Spaces</p>
            </div>
          </div>

          <button
            onClick={() => setDarkMode(!darkMode)}
            className="p-2 rounded-lg hover:bg-slate-800 transition-colors"
          >
            {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          </button>
        </div>
      </header>

      {/* === Main Content === */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-6">
          {/* === Canvas Area === */}
          <div className="space-y-4">
            {/* Side-by-side comparison view */}
            {state.showComparison && state.renderedImage ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-slate-300">Before / After Comparison</h3>
                  <button
                    onClick={() => setState(prev => ({ ...prev, showComparison: false, renderedImage: null }))}
                    className="px-3 py-1 text-sm bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
                  >
                    ✕ Close
                  </button>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <div className="text-center text-sm text-slate-400 font-medium">Before</div>
                    <div className="rounded-xl overflow-hidden border-2 border-slate-700">
                      <img
                        src={state.image || ''}
                        alt="Original layout"
                        className="w-full h-auto"
                      />
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="text-center text-sm text-emerald-400 font-medium">After</div>
                    <div className="rounded-xl overflow-hidden border-2 border-emerald-600">
                      <img
                        src={state.renderedImage}
                        alt="Optimized layout"
                        className="w-full h-auto"
                      />
                    </div>
                  </div>
                </div>
              </div>
            ) : state.image && state.objects.length > 0 ? (
              <CanvasOverlay
                imageUrl={state.image}
                objects={state.objects}
                selectedObjectId={state.selectedObjectId}
                lockedObjectIds={state.lockedObjectIds}
                overlays={state.overlays}
                onObjectSelect={handleObjectSelect}
                onObjectLock={handleObjectLock}
                maskMode={state.maskMode}
                onMaskComplete={handleMaskComplete}
              />
            ) : (
              <ImageUpload
                onImageSelect={handleImageSelect}
                currentImage={state.image}
                disabled={state.isAnalyzing}
              />
            )}

            {/* Room dimensions */}
            {state.roomDimensions && (
              <div className="text-center text-sm text-slate-500">
                Room: {state.roomDimensions.width_estimate}px × {state.roomDimensions.height_estimate}px
              </div>
            )}
          </div>

          {/* === Control Panel === */}
          <ControlPanel
            state={state}
            onAnalyze={handleAnalyze}
            onOptimize={handleOptimize}
            onObjectSelect={handleObjectSelect}
            onObjectLock={handleObjectLock}
            onMaskModeToggle={handleMaskModeToggle}
            onAddEdit={handleAddEdit}
            onApplyEdits={handleApplyEdits}
            onRemoveEdit={handleRemoveEdit}
          />
        </div>
      </main>

      {/* === Footer === */}
      <footer className="border-t border-slate-800 bg-slate-900/50 mt-8">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between text-sm text-slate-500">
            <div className="flex items-center gap-4">
              {state.violations.length > 0 ? (
                <span className="flex items-center gap-2 text-amber-400">
                  ⚠️ {state.violations.length} issue(s) detected
                </span>
              ) : state.objects.length > 0 ? (
                <span className="flex items-center gap-2 text-emerald-400">
                  ✅ All constraints satisfied
                </span>
              ) : (
                <span>Upload a bedroom image to get started</span>
              )}
            </div>
            <div>
              {state.objects.length > 0 && (
                <span>{state.objects.filter(o => o.type === 'movable').length} movable objects</span>
              )}
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
