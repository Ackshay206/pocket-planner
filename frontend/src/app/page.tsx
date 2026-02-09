'use client';

import { useState, useCallback, useEffect } from 'react';
import { Toaster, toast } from 'react-hot-toast';
import { ImageUpload } from '@/components/ImageUpload';
import { CanvasOverlay } from '@/components/CanvasOverlay';
import { ObjectsPanel } from '@/components/ObjectsPanel';
import { OptimizePanel } from '@/components/OptimizePanel';
import { LayoutSelector } from '@/components/LayoutSelector';
import { PerspectiveView } from '@/components/PerspectiveView';
import { ChatEditor } from '@/components/ChatEditor';
import { ProductRecommendations } from '@/components/ProductRecommendations';
import { GeneratingOverlay } from '@/components/GeneratingOverlay';
import { useAnalyze } from '@/hooks/useAnalyze';
import { useOptimize } from '@/hooks/useOptimize';
import { usePerspective } from '@/hooks/usePerspective';
import { useChatEdit } from '@/hooks/useChatEdit';
import { useShop } from '@/hooks/useShop';
import { Layout, Maximize2, Sparkles } from 'lucide-react';
import type { RoomObject, RoomDimensions, LayoutVariation, AppStage } from '@/lib/types';

interface AppState {
  stage: AppStage;
  image: string | null;
  roomDimensions: RoomDimensions | null;
  objects: RoomObject[];
  selectedObjectId: string | null;
  isAnalyzing: boolean;
  layoutVariations: LayoutVariation[];
  selectedVariation: LayoutVariation | null;
  perspectiveImage: string | null;
  currentLayout: RoomObject[];
  generatingStep: number;
  shopBudget: number;
}

const initialState: AppState = {
  stage: 'analyze',
  image: null,
  roomDimensions: null,
  objects: [],
  selectedObjectId: null,
  isAnalyzing: false,
  layoutVariations: [],
  selectedVariation: null,
  perspectiveImage: null,
  currentLayout: [],
  generatingStep: 0,
  shopBudget: 1000,
};

// Generation steps for the loading animation
const GENERATION_STEPS = [
  { label: 'Analyzing room layout', duration: 2000 },
  { label: 'Designing Productivity Focus', duration: 3000 },
  { label: 'Designing Cozy Retreat', duration: 3000 },
  { label: 'Designing Space Optimized', duration: 3000 },
  { label: 'Generating preview images', duration: 4000 },
  { label: 'Finalizing layouts', duration: 2000 },
];

export default function DwellAiApp() {
  const [state, setState] = useState<AppState>(initialState);

  const { analyze } = useAnalyze();
  const { optimize, isLoading: isOptimizing } = useOptimize();
  const { generate: generatePerspective, isLoading: isGeneratingPerspective } = usePerspective();
  const { sendCommand, isLoading: isChatLoading, messages } = useChatEdit();
  const { findProducts, isLoading: isShopping, data: shopData } = useShop();

  // Auto-load test image on mount
  useEffect(() => {
    const loadTestImage = async () => {
      try {
        const response = await fetch('/test_img.jpg');
        const blob = await response.blob();
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64 = reader.result as string;
          // Only auto-load if no image is present? 
          // Actually user said "make image analysis the first page". 
          // Auto-loading test image might bypass the "Upload" hero view.
          // Let's Comment this out to force the "First Page" experience or keep it but checking constraints.
          // setState(prev => ({ ...prev, image: base64 }));
        };
        // reader.readAsDataURL(blob);
      } catch (error) {
        console.log('No test image found');
      }
    };
    // loadTestImage();
  }, []);

  // Step through generation animation
  useEffect(() => {
    if (!isOptimizing) {
      setState(prev => ({ ...prev, generatingStep: 0 }));
      return;
    }

    const stepInterval = setInterval(() => {
      setState(prev => {
        if (prev.generatingStep < GENERATION_STEPS.length - 1) {
          return { ...prev, generatingStep: prev.generatingStep + 1 };
        }
        return prev;
      });
    }, 2500);

    return () => clearInterval(stepInterval);
  }, [isOptimizing]);

  // Image Upload
  const handleImageSelect = useCallback((base64: string) => {
    setState(prev => ({
      ...prev,
      stage: 'analyze',
      image: `data:image/jpeg;base64,${base64}`,
      objects: [],
      selectedObjectId: null,
      layoutVariations: [],
      selectedVariation: null,
      perspectiveImage: null,
    }));
  }, []);

  // Analyze
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
        isAnalyzing: false,
        stage: 'analyze',
      }));

      toast.success(`Detected ${response.objects.length} objects`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Analysis failed');
      setState(prev => ({ ...prev, isAnalyzing: false }));
    }
  }, [state.image, analyze]);

  // Generate Layouts
  const handleGenerateLayouts = useCallback(async () => {
    if (!state.roomDimensions || state.objects.length === 0) return;

    try {
      const lockedIds = state.objects
        .filter(o => o.is_locked || o.type === 'structural')
        .map(o => o.id);

      const response = await optimize({
        current_layout: state.objects,
        locked_ids: lockedIds,
        room_dimensions: state.roomDimensions,
        image_base64: state.image ? state.image.split(',')[1] : undefined,
      });

      setState(prev => ({
        ...prev,
        layoutVariations: response.variations,
        stage: 'layouts',
      }));

      toast.success(`Generated ${response.variations.length} layout options`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Failed to generate layouts');
    }
  }, [state.roomDimensions, state.objects, optimize, state.image]);

  // Select Layout
  const handleSelectLayout = useCallback(async (variation: LayoutVariation) => {
    setState(prev => ({
      ...prev,
      selectedVariation: variation,
      currentLayout: variation.layout,
      stage: 'perspective',
    }));

    if (state.roomDimensions) {
      try {
        const response = await generatePerspective({
          layout: variation.layout,
          room_dimensions: state.roomDimensions,
          style: 'modern',
          view_angle: 'entrance',
          image_base64: variation.thumbnail_base64 || undefined,
          layout_plan: variation.layout_plan || undefined,
          door_info: variation.door_info || undefined,
          window_info: variation.window_info || undefined,
        });

        setState(prev => ({
          ...prev,
          perspectiveImage: response.image_base64,
        }));

        toast.success('Perspective view generated!');
      } catch (error) {
        toast.error(error instanceof Error ? error.message : 'Failed to generate perspective');
      }
    }
  }, [state.roomDimensions, generatePerspective]);

  // Object selection
  const handleObjectSelect = useCallback((id: string) => {
    setState(prev => ({
      ...prev,
      selectedObjectId: prev.selectedObjectId === id ? null : id,
    }));
  }, []);

  // Navigation handlers
  const handleBackToAnalyze = useCallback(() => {
    setState(prev => ({ ...prev, stage: 'analyze', layoutVariations: [] }));
  }, []);

  const handleBackToLayouts = useCallback(() => {
    setState(prev => ({ ...prev, stage: 'layouts' }));
  }, []);

  const handleBackToPerspective = useCallback(() => {
    setState(prev => ({ ...prev, stage: 'perspective' }));
  }, []);

  const handleContinueToChat = useCallback(() => {
    setState(prev => ({ ...prev, stage: 'chat' }));
  }, []);

  const handleChatCommand = useCallback(async (command: string) => {
    if (!state.roomDimensions || !state.currentLayout.length) return;

    try {
      // Filter to structural only to avoid confusing the model with stale furniture
      const structuralLayout = state.currentLayout.filter(o => o.type === 'structural');
      const movableLayout = state.currentLayout.filter(o => o.type === 'movable');

      const response = await sendCommand({
        command,
        current_layout: structuralLayout,
        room_dimensions: state.roomDimensions,
        current_image_base64: state.perspectiveImage || undefined,
        layout_plan: state.selectedVariation?.layout_plan || undefined,
      });

      if (response.updated_layout) {
        // Merge the updated structural layout back with the preserved furniture
        setState(prev => ({
          ...prev,
          currentLayout: [...response.updated_layout, ...movableLayout],
        }));
      }

      if (response.updated_image_base64) {
        setState(prev => ({
          ...prev,
          perspectiveImage: response.updated_image_base64!,
        }));
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Edit failed');
    }
  }, [state.roomDimensions, state.currentLayout, state.perspectiveImage, sendCommand]);

  // Reset
  const handleContinueToShop = useCallback(() => {
    setState(prev => ({ ...prev, stage: 'shop' }));
  }, []);

  const handleBackToChat = useCallback(() => {
    setState(prev => ({ ...prev, stage: 'chat' }));
  }, []);

  const handleShopSearch = useCallback(async () => {
    if (!state.currentLayout.length) return;
    try {
      await findProducts({
        current_layout: state.currentLayout,
        total_budget: state.shopBudget,
        perspective_image_base64: state.perspectiveImage || undefined,
      });
      toast.success('Products found!');
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Product search failed');
    }
  }, [state.currentLayout, state.shopBudget, state.perspectiveImage, findProducts]);

  // Reset
  const handleReset = useCallback(() => {
    setState(initialState);
  }, []);

  // Render based on stage
  const renderContent = () => {
    if (isOptimizing) {
      return (
        <GeneratingOverlay
          steps={GENERATION_STEPS}
          currentStep={state.generatingStep}
        />
      );
    }

    switch (state.stage) {
      case 'layouts':
        return (
          <LayoutSelector
            variations={state.layoutVariations}
            roomDimensions={state.roomDimensions}
            isLoading={false}
            onSelect={handleSelectLayout}
            onBack={handleBackToAnalyze}
          />
        );

      case 'perspective':
        return (
          <PerspectiveView
            imageBase64={state.perspectiveImage}
            isLoading={isGeneratingPerspective}
            layoutName={state.selectedVariation?.name}
            onContinue={handleContinueToChat}
            onBack={handleBackToLayouts}
          />
        );

      case 'shop':
        return (
          <ProductRecommendations
            shopData={shopData}
            isLoading={isShopping}
            budget={state.shopBudget}
            onBudgetChange={(b) => setState(prev => ({ ...prev, shopBudget: b }))}
            onSearch={handleShopSearch}
            onBack={handleBackToChat}
            hasLayout={state.currentLayout.length > 0}
          />
        );

      case 'chat':
        return (
          <div className="space-y-4">
            <ChatEditor
              imageBase64={state.perspectiveImage}
              layout={state.currentLayout}
              roomDimensions={state.roomDimensions}
              messages={messages}
              isLoading={isChatLoading}
              onSendCommand={handleChatCommand}
              onBack={handleBackToPerspective}
            />
            <div className="text-center pt-4 border-t border-gray-100">
              <button
                onClick={handleContinueToShop}
                className="px-6 py-3 bg-black text-white font-medium hover:bg-gray-800 transition-colors inline-flex items-center gap-2 rounded-full"
              >
                <span>🛒</span>
                Shop Your Room
              </button>
            </div>
          </div>
        );

      case 'analyze':
      default:
        // Hero / Upload State
        if (!state.image) {
          return (
            <div className="max-w-5xl mx-auto px-6 py-12">
              <div className="text-center mb-16 space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-700">
                <h1 className="text-5xl md:text-6xl font-bold tracking-tighter text-black mb-6">
                  Interior Design <br />
                  <span className="text-gray-400 font-light italic">Reimagined by AI</span>
                </h1>
                <p className="text-lg text-gray-500 max-w-xl mx-auto leading-relaxed">
                  Upload your floor plan and let our AI generate optimized, beautiful styling options instantly.
                </p>
              </div>

              <div className="max-w-2xl mx-auto shadow-2xl shadow-gray-200/50 rounded-3xl overflow-hidden animate-in fade-in zoom-in-95 duration-700 delay-150">
                <ImageUpload onImageSelect={handleImageSelect} />
              </div>

              <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-8 text-center opacity-60">
                <div>
                  <Layout className="w-6 h-6 mx-auto mb-3" />
                  <h3 className="font-semibold mb-1">Smart Analysis</h3>
                  <p className="text-sm text-gray-400">Instantly detects structure & furniture</p>
                </div>
                <div>
                  <Sparkles className="w-6 h-6 mx-auto mb-3" />
                  <h3 className="font-semibold mb-1">AI Design</h3>
                  <p className="text-sm text-gray-400">Generates 3 unique style variations</p>
                </div>
                <div>
                  <Maximize2 className="w-6 h-6 mx-auto mb-3" />
                  <h3 className="font-semibold mb-1">Space Optimization</h3>
                  <p className="text-sm text-gray-400">Maximizes flow and functionality</p>
                </div>
              </div>
            </div>
          );
        }

        // Split view for Analysis/Optimizing
        return (
          <div className="flex flex-col gap-6">
            <div className="flex flex-col lg:flex-row gap-6">
              {/* Floor Plan Viewer */}
              <div className="flex-1 min-w-0">
                {state.image && state.objects.length > 0 ? (
                  <CanvasOverlay
                    imageUrl={state.image}
                    objects={state.objects}
                    selectedObjectId={state.selectedObjectId}
                    onObjectSelect={handleObjectSelect}
                  />
                ) : (
                  // analyzing state visualization
                  <div className="floor-plan-container p-4">
                    <div className="relative overflow-hidden rounded-lg inline-block w-full">
                      <img
                        src={state.image}
                        alt="Floor plan"
                        className="w-full h-auto rounded-none"
                      />
                      {/* Scanning Animation */}
                      {state.isAnalyzing && (
                        <div className="animate-scan"></div>
                      )}
                    </div>
                    <div className="text-center mt-6">
                      <button
                        onClick={handleAnalyze}
                        disabled={state.isAnalyzing}
                        className="bg-black text-white px-8 py-3 font-medium hover:bg-gray-800 transition-colors disabled:opacity-50 rounded-full"
                      >
                        {state.isAnalyzing ? 'Analyzing...' : 'Analyze Room'}
                      </button>
                    </div>
                  </div>
                )}
              </div>

              {/* Right Panel - Optimize */}
              {state.objects.length > 0 && (
                <div className="w-full lg:w-80 shrink-0">
                  <OptimizePanel
                    objects={state.objects}
                    roomDimensions={state.roomDimensions}
                    isGenerating={isOptimizing}
                    onGenerate={handleGenerateLayouts}
                    onReanalyze={handleAnalyze}
                    isAnalyzing={state.isAnalyzing}
                  />
                </div>
              )}
            </div>

            {/* Detected Objects Panel - Below the image */}
            {state.objects.length > 0 && (
              <ObjectsPanel
                objects={state.objects}
                selectedObjectId={state.selectedObjectId}
                onObjectSelect={handleObjectSelect}
              />
            )}

            {/* Room dimensions */}
            {state.roomDimensions && (
              <div className="text-center text-sm text-gray-400">
                Room: {state.roomDimensions.width_estimate.toFixed(1)} × {state.roomDimensions.height_estimate.toFixed(1)} ft
              </div>
            )}
          </div>
        );
    }
  };

  return (
    <div className="min-h-screen bg-white">
      <Toaster
        position="top-right"
        toastOptions={{
          className: '!bg-white !text-gray-800 !shadow-lg',
          style: { borderRadius: '0', border: '1px solid #e5e5e5' },
        }}
      />

      {/* Header */}
      <header className="py-6 border-b border-gray-100 mb-8 sticky top-0 bg-white/80 backdrop-blur z-50">
        <div className="max-w-7xl mx-auto px-4 flex justify-between items-center">
          <div className="flex items-center gap-2 cursor-pointer" onClick={handleReset}>
            <div className="w-8 h-8 bg-black flex items-center justify-center rounded-lg">
              <span className="text-white font-serif font-bold text-lg">D</span>
            </div>
            <span className="font-bold text-xl tracking-tight">Dwell.ai</span>
          </div>

          {/* Stage indicator - only show if image uploaded */}
          {state.image && (
            <div className="flex gap-2">
              {['analyze', 'layouts', 'perspective', 'chat', 'shop'].map((s) => (
                <div
                  key={s}
                  className={`w-2 h-2 rounded-full transition-colors ${state.stage === s ? 'bg-black' : 'bg-gray-200'
                    }`}
                />
              ))}
            </div>
          )}
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 pb-8">
        <div className={state.image ? "bg-white p-6 border border-gray-100 rounded-2xl shadow-sm" : ""}>
          {renderContent()}
        </div>
      </main>
    </div>
  );
}