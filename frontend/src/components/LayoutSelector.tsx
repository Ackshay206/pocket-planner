'use client';

import { ArrowLeft, Loader2, Check, ImageOff } from 'lucide-react';
import { useState } from 'react';
import type { LayoutVariation, RoomDimensions, RoomObject } from '@/lib/types';

interface LayoutSelectorProps {
    variations: LayoutVariation[];
    roomDimensions: RoomDimensions | null;
    isLoading: boolean;
    onSelect: (variation: LayoutVariation) => void;
    onBack: () => void;
}

// Color mapping for furniture types - Neutral / Earthy tones
const getObjectColor = (label: string): string => {
    const labelLower = label.toLowerCase();
    if (labelLower.includes('bed')) return '#D7CCC8'; // Beige/Taupe
    if (labelLower.includes('sofa') || labelLower.includes('couch')) return '#EFEBE9'; // Light Grey/Beige
    if (labelLower.includes('chair')) return '#CFD8DC'; // Blue Grey
    if (labelLower.includes('desk')) return '#BCAAA4'; // Brownish
    if (labelLower.includes('table')) return '#D7CCC8';
    if (labelLower.includes('door')) return '#90A4AE';
    if (labelLower.includes('window')) return '#B0BEC5';
    if (labelLower.includes('wardrobe') || labelLower.includes('closet')) return '#A1887F';
    return '#E0E0E0';
};

// SVG-based preview component (fallback when no thumbnail)
function LayoutPreview({ layout, roomDimensions }: { layout: RoomObject[]; roomDimensions: RoomDimensions | null }) {
    if (!layout || layout.length === 0) {
        return (
            <div className="w-full h-full flex items-center justify-center">
                <span className="text-xs text-gray-400">No preview</span>
            </div>
        );
    }

    // Calculate bounds
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    layout.forEach(obj => {
        const [x, y, w, h] = obj.bbox;
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x + w);
        maxY = Math.max(maxY, y + h);
    });

    const padding = Math.max((maxX - minX), (maxY - minY)) * 0.1;
    const viewX = minX - padding;
    const viewY = minY - padding;
    const viewW = (maxX - minX) + (padding * 2);
    const viewH = (maxY - minY) + (padding * 2);

    return (
        <svg
            viewBox={`${viewX} ${viewY} ${viewW} ${viewH}`}
            className="w-full h-full"
            preserveAspectRatio="xMidYMid meet"
        >
            {layout.map((obj, i) => {
                const [x, y, w, h] = obj.bbox;
                const color = getObjectColor(obj.label);
                const isStructural = obj.type === 'structural';

                return (
                    <g key={obj.id || i}>
                        <rect
                            x={x} y={y} width={w} height={h}
                            fill={color}
                            stroke={isStructural ? '#424242' : '#757575'}
                            strokeWidth={Math.max(1, viewW * 0.005)}
                            strokeDasharray={isStructural ? `${viewW * 0.01},${viewW * 0.005}` : 'none'}
                            rx={viewW * 0.01}
                            fillOpacity={0.9}
                        />
                    </g>
                );
            })}
        </svg>
    );
}

// Thumbnail image component with error handling
function ThumbnailImage({ base64, name, onError }: { base64: string; name: string; onError: () => void }) {
    const [loaded, setLoaded] = useState(false);
    const [error, setError] = useState(false);

    // Try different image formats
    const imageSrc = base64.startsWith('data:')
        ? base64
        : `data:image/png;base64,${base64}`;

    if (error) {
        return null; // Will trigger fallback to SVG preview
    }

    return (
        <>
            {!loaded && (
                <div className="absolute inset-0 flex items-center justify-center bg-gray-50">
                    <Loader2 className="w-6 h-6 animate-spin text-gray-300" />
                </div>
            )}
            <img
                src={imageSrc}
                alt={name}
                className={`w-full h-full object-contain mix-blend-multiply transition-opacity duration-500 ${loaded ? 'opacity-100' : 'opacity-0'}`}
                onLoad={() => setLoaded(true)}
                onError={() => {
                    console.warn(`Failed to load thumbnail for ${name}`);
                    setError(true);
                    onError();
                }}
            />
        </>
    );
}

export function LayoutSelector({
    variations,
    roomDimensions,
    isLoading,
    onSelect,
    onBack,
}: LayoutSelectorProps) {
    const [failedThumbnails, setFailedThumbnails] = useState<Set<number>>(new Set());

    if (isLoading) {
        return (
            <div className="flex flex-col items-center justify-center min-h-[400px] gap-4">
                <Loader2 className="w-12 h-12 animate-spin text-black" />
                <p className="text-black text-lg font-medium tracking-tight">Generating layout variations...</p>
                <p className="text-gray-400 text-sm">Our AI is creating 3 unique designs for you</p>
            </div>
        );
    }

    if (!variations || variations.length === 0) {
        return (
            <div className="flex flex-col items-center justify-center min-h-[400px] gap-4">
                <ImageOff className="w-12 h-12 text-gray-300" />
                <p className="text-gray-900 text-lg">No layouts generated</p>
                <button
                    onClick={onBack}
                    className="px-6 py-2 bg-black text-white rounded-full hover:bg-gray-800 transition-colors"
                >
                    Go Back
                </button>
            </div>
        );
    }

    return (
        <div className="space-y-8 animate-in fade-in duration-500">
            {/* Header */}
            <div className="flex items-center gap-4 border-b border-gray-100 pb-6">
                <button
                    onClick={onBack}
                    className="p-2 rounded-full hover:bg-gray-100 transition-colors"
                >
                    <ArrowLeft className="w-5 h-5 text-black" />
                </button>
                <div>
                    <h2 className="text-2xl font-bold text-black tracking-tight">Choose Your Layout</h2>
                    <p className="text-sm text-gray-400">Select one of the AI-generated options</p>
                </div>
            </div>

            {/* Layout Cards */}
            <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3">
                {variations.map((variation, index) => {
                    const hasThumbnail = variation.thumbnail_base64 && !failedThumbnails.has(index);

                    // Summarize unique furniture types
                    const uniqueFurniture = Array.from(new Set(
                        variation.layout
                            .filter(obj => obj.type === 'movable')
                            .map(obj => obj.label.split('_')[0].toLowerCase())
                    )).filter(label => !['rug', 'plant', 'lamp', 'artwork'].includes(label));

                    // Capitalize first letter
                    const furnitureList = uniqueFurniture.map(l => l.charAt(0).toUpperCase() + l.slice(1));

                    return (
                        <div
                            key={index}
                            className="group cursor-pointer flex flex-col gap-4 border border-gray-100 rounded-2xl p-4 hover:border-black transition-all shadow-sm hover:shadow-md"
                            onClick={() => onSelect(variation)}
                        >
                            {/* Layout Preview Card */}
                            <div className="aspect-square bg-gray-50 rounded-xl relative overflow-hidden border border-gray-100">
                                {/* Thumbnail or SVG Preview */}
                                <div className="absolute inset-0 p-6">
                                    {hasThumbnail ? (
                                        <ThumbnailImage
                                            base64={variation.thumbnail_base64!}
                                            name={variation.name}
                                            onError={() => {
                                                setFailedThumbnails(prev => new Set(prev).add(index));
                                            }}
                                        />
                                    ) : (
                                        <LayoutPreview
                                            layout={variation.layout}
                                            roomDimensions={roomDimensions}
                                        />
                                    )}
                                </div>

                                {/* Hover Overlay */}
                                <div className="absolute inset-0 bg-black/5 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                                    <div className="bg-black text-white px-6 py-2 rounded-full font-medium transform translate-y-4 group-hover:translate-y-0 transition-transform">
                                        View in 3D
                                    </div>
                                </div>
                            </div>

                            {/* Layout Info */}
                            <div className="space-y-3">
                                <div>
                                    <h3 className="font-bold text-xl text-black mb-1">{variation.name}</h3>
                                    <p className="text-sm text-gray-500 leading-relaxed min-h-[40px]">{variation.description}</p>
                                </div>

                                {/* Key Furniture Items */}
                                <div className="flex flex-wrap gap-2 pt-2 border-t border-gray-100">
                                    {furnitureList.slice(0, 4).map((item, i) => (
                                        <span key={i} className="text-xs font-medium bg-gray-100 text-gray-600 px-2.5 py-1 rounded-full">
                                            {item}
                                        </span>
                                    ))}
                                    {furnitureList.length > 4 && (
                                        <span className="text-xs font-medium text-gray-400 px-1 py-1">
                                            +{furnitureList.length - 4} more
                                        </span>
                                    )}
                                </div>

                                {/* Stats */}
                                <div className="flex items-center gap-4 text-xs text-gray-400 pt-1">
                                    <span className="flex items-center gap-1">
                                        <div className="w-1.5 h-1.5 rounded-full bg-green-400" />
                                        {variation.layout.filter(o => o.type === 'movable').length} Items
                                    </span>
                                    {variation.layout_plan && (
                                        <span className="flex items-center gap-1">
                                            <div className="w-1.5 h-1.5 rounded-full bg-blue-400" />
                                            AI Planned
                                        </span>
                                    )}
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}