'use client';

import { ArrowLeft, Loader2, MessageSquare } from 'lucide-react';

interface PerspectiveViewProps {
    imageBase64: string | null;
    isLoading: boolean;
    layoutName?: string;
    onContinue: () => void;
    onBack: () => void;
}

export function PerspectiveView({
    imageBase64,
    isLoading,
    layoutName,
    onContinue,
    onBack,
}: PerspectiveViewProps) {
    if (isLoading) {
        return (
            <div className="flex flex-col items-center justify-center min-h-[500px] gap-4">
                <Loader2 className="w-12 h-12 animate-spin text-[#6b7aa1]" />
                <p className="text-gray-500 text-lg">Generating perspective view...</p>
                <p className="text-gray-400 text-sm">Creating a photorealistic render of your layout</p>
            </div>
        );
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <button
                        onClick={onBack}
                        className="p-2 bg-black text-white rounded-full hover:bg-gray-800 transition-colors"
                    >
                        <ArrowLeft className="w-5 h-5" />
                    </button>
                    <div>
                        <h2 className="text-xl font-semibold text-gray-800">
                            {layoutName || 'Your Layout'}
                        </h2>
                        <p className="text-sm text-gray-500">Perspective view generated</p>
                    </div>
                </div>

                <button
                    onClick={onContinue}
                    className="flex items-center gap-2 px-4 py-2 bg-black text-white rounded-full font-medium hover:bg-gray-800 transition-colors"
                >
                    <MessageSquare className="w-4 h-4" />
                    Edit with Chat
                </button>
            </div>

            {/* Image Display */}
            <div className="floor-plan-container">
                {imageBase64 ? (
                    <img
                        src={`data:image/png;base64,${imageBase64}`}
                        alt="Perspective view"
                        className="w-full h-auto rounded-xl shadow-lg"
                    />
                ) : (
                    <div className="aspect-video bg-gradient-to-br from-[#f5f3f0] to-[#e8e6e3] rounded-xl flex items-center justify-center">
                        <p className="text-gray-400">No perspective image available</p>
                    </div>
                )}
            </div>

            {/* Action Hint */}
            <div className="text-center">
                <p className="text-sm text-gray-500">
                    Click "Edit with Chat" to make conversational changes to your room
                </p>
            </div>
        </div>
    );
}
