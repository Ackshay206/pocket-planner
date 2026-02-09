'use client';

import { Sparkles, RefreshCw, Loader2, Wand2, Layout, Palette, Zap } from 'lucide-react';
import type { RoomObject, RoomDimensions } from '@/lib/types';

interface OptimizePanelProps {
    objects: RoomObject[];
    roomDimensions: RoomDimensions | null;
    isGenerating: boolean;
    onGenerate: () => void;
    onReanalyze: () => void;
    isAnalyzing: boolean;
}

export function OptimizePanel({
    objects,
    roomDimensions,
    isGenerating,
    onGenerate,
    onReanalyze,
    isAnalyzing,
}: OptimizePanelProps) {
    const movableCount = objects.filter(o => o.type === 'movable').length;
    const structuralCount = objects.filter(o => o.type === 'structural').length;

    return (
        <div className="bg-white border border-gray-200 h-full flex flex-col p-6">
            {/* Header */}
            <div className="text-center mb-8">
                <h2 className="text-xl font-bold text-black tracking-tight mb-2">AI Layout Designer</h2>
                <div className="h-px w-12 bg-black mx-auto"></div>
                <p className="text-sm text-gray-500 mt-3">Generate optimized room arrangements</p>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-2 gap-4 mb-8">
                <div className="border border-gray-100 p-4 text-center">
                    <div className="text-3xl font-light text-black">{movableCount}</div>
                    <div className="text-xs text-gray-400 uppercase tracking-widest mt-1">Movable</div>
                </div>
                <div className="border border-gray-100 p-4 text-center">
                    <div className="text-3xl font-light text-black">{structuralCount}</div>
                    <div className="text-xs text-gray-400 uppercase tracking-widest mt-1">Fixed</div>
                </div>
            </div>

            {/* Layout Styles Preview */}
            <div className="mb-8">
                <div className="text-xs font-semibold text-black uppercase tracking-widest mb-4">
                    Styles
                </div>
                <div className="space-y-3">
                    <div className="flex items-center gap-4 p-3 border border-gray-100 hover:border-black transition-colors group cursor-default">
                        <div className="w-8 h-8 flex items-center justify-center bg-gray-50 group-hover:bg-black transition-colors">
                            <Zap className="w-4 h-4 text-gray-400 group-hover:text-white" />
                        </div>
                        <div>
                            <div className="text-sm font-medium text-black">Productivity Focus</div>
                            <div className="text-xs text-gray-400">Workstation optimized</div>
                        </div>
                    </div>
                    <div className="flex items-center gap-4 p-3 border border-gray-100 hover:border-black transition-colors group cursor-default">
                        <div className="w-8 h-8 flex items-center justify-center bg-gray-50 group-hover:bg-black transition-colors">
                            <Layout className="w-4 h-4 text-gray-400 group-hover:text-white" />
                        </div>
                        <div>
                            <div className="text-sm font-medium text-black">Cozy Retreat</div>
                            <div className="text-xs text-gray-400">Relaxation & Comfort</div>
                        </div>
                    </div>
                    <div className="flex items-center gap-4 p-3 border border-gray-100 hover:border-black transition-colors group cursor-default">
                        <div className="w-8 h-8 flex items-center justify-center bg-gray-50 group-hover:bg-black transition-colors">
                            <Palette className="w-4 h-4 text-gray-400 group-hover:text-white" />
                        </div>
                        <div>
                            <div className="text-sm font-medium text-black">Space Optimized</div>
                            <div className="text-xs text-gray-400">Maximize floor space</div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Spacer */}
            <div className="flex-1" />

            {/* Generate Button */}
            <button
                onClick={onGenerate}
                disabled={isGenerating || movableCount === 0}
                className="w-full py-4 text-sm font-medium tracking-wide bg-black text-white 
          hover:bg-gray-800 transition-all disabled:opacity-50 disabled:cursor-not-allowed
          flex items-center justify-center gap-2 rounded-full"
            >
                {isGenerating ? (
                    <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        GENERATING...
                    </>
                ) : (
                    <>
                        GENERATE LAYOUTS
                    </>
                )}
            </button>

            {/* Re-analyze button */}
            <button
                onClick={onReanalyze}
                disabled={isAnalyzing}
                className="w-full mt-3 py-3 text-xs font-medium tracking-wide text-gray-400 hover:text-black 
          transition-colors flex items-center justify-center gap-2 rounded-full"
            >
                {isAnalyzing ? (
                    <>
                        <Loader2 className="w-3 h-3 animate-spin" />
                        RE-ANALYZING...
                    </>
                ) : (
                    <>
                        <RefreshCw className="w-3 h-3" />
                        RE-ANALYZE ROOM
                    </>
                )}
            </button>

            {/* Info */}
            {movableCount === 0 && (
                <p className="text-xs text-gray-400 text-center mt-4">
                    No movable objects. Add furniture to generate layouts.
                </p>
            )}
        </div>
    );
}