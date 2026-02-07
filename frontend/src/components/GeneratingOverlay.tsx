'use client';

import { useEffect, useState } from 'react';
import { Sparkles, Check, Loader2 } from 'lucide-react';

interface GeneratingOverlayProps {
    steps: Array<{ label: string; duration: number }>;
    currentStep: number;
}

export function GeneratingOverlay({ steps, currentStep }: GeneratingOverlayProps) {
    const [dots, setDots] = useState('');

    // Animate dots
    useEffect(() => {
        const interval = setInterval(() => {
            setDots(prev => (prev.length >= 3 ? '' : prev + '.'));
        }, 400);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="min-h-[500px] flex flex-col items-center justify-center p-8 bg-white/95 backdrop-blur-xl rounded-3xl border border-gray-100 shadow-2xl">
            {/* Animated Icon */}
            <div className="relative mb-8">
                <div className="w-24 h-24 bg-black rounded-full flex items-center justify-center shadow-2xl animate-pulse">
                    <Sparkles className="w-10 h-10 text-white" />
                </div>
                {/* Orbiting dots - Beige */}
                <div className="absolute inset-0 animate-spin" style={{ animationDuration: '3s' }}>
                    <div className="absolute -top-2 left-1/2 w-3 h-3 bg-[#C4A484] rounded-full transform -translate-x-1/2" />
                </div>
                <div className="absolute inset-0 animate-spin" style={{ animationDuration: '4s', animationDirection: 'reverse' }}>
                    <div className="absolute -bottom-2 left-1/2 w-2 h-2 bg-gray-400 rounded-full transform -translate-x-1/2" />
                </div>
            </div>

            {/* Title */}
            <h2 className="text-2xl font-bold text-black mb-2 tracking-tight">
                Designing Your Space{dots}
            </h2>
            <p className="text-gray-500 mb-8 font-light">Our AI is creating luxury layout options</p>

            {/* Progress Steps */}
            <div className="w-full max-w-md space-y-4">
                {steps.map((step, index) => {
                    const isComplete = index < currentStep;
                    const isCurrent = index === currentStep;

                    return (
                        <div
                            key={index}
                            className={`
                                flex items-center gap-4 p-4 rounded-xl transition-all duration-500 border
                                ${isComplete ? 'bg-black border-black' : isCurrent ? 'bg-white border-gray-200 shadow-lg scale-105' : 'bg-gray-50 border-transparent opacity-50'}
                            `}
                        >
                            {/* Status Icon */}
                            <div className={`
                                w-6 h-6 rounded-full flex items-center justify-center shrink-0 transition-all border
                                ${isComplete
                                    ? 'bg-white border-white text-black'
                                    : isCurrent
                                        ? 'bg-black border-black text-white'
                                        : 'bg-transparent border-gray-300 text-gray-300'
                                }
                            `}>
                                {isComplete ? (
                                    <Check className="w-3 h-3" />
                                ) : isCurrent ? (
                                    <Loader2 className="w-3 h-3 animate-spin" />
                                ) : (
                                    <span className="text-[10px] font-medium">{index + 1}</span>
                                )}
                            </div>

                            {/* Label */}
                            <span className={`
                                text-sm font-medium transition-colors
                                ${isComplete
                                    ? 'text-white'
                                    : isCurrent
                                        ? 'text-black'
                                        : 'text-gray-400'
                                }
                            `}>
                                {step.label}
                            </span>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}