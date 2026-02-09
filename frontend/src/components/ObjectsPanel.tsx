'use client';

import { type RoomObject, getObjectIcon } from '@/lib/types';

interface ObjectsPanelProps {
    objects: RoomObject[];
    selectedObjectId: string | null;
    onObjectSelect: (id: string) => void;
}

const getObjectStyle = (label: string): { bg: string; border: string; text: string } => {
    const l = label.toLowerCase();
    if (l.includes('bed')) return { bg: 'bg-rose-50', border: 'border-rose-200', text: 'text-rose-700' };
    if (l.includes('sofa') || l.includes('couch')) return { bg: 'bg-pink-50', border: 'border-pink-200', text: 'text-pink-700' };
    if (l.includes('chair') || l.includes('desk')) return { bg: 'bg-emerald-50', border: 'border-emerald-200', text: 'text-emerald-700' };
    if (l.includes('table')) return { bg: 'bg-amber-50', border: 'border-amber-200', text: 'text-amber-700' };
    if (l.includes('door') || l.includes('window')) return { bg: 'bg-slate-50', border: 'border-slate-200', text: 'text-slate-600' };
    if (l.includes('wardrobe') || l.includes('closet')) return { bg: 'bg-orange-50', border: 'border-orange-200', text: 'text-orange-700' };
    return { bg: 'bg-gray-50', border: 'border-gray-200', text: 'text-gray-600' };
};

const formatLabel = (label: string): string => {
    return label.split('_')[0].replace(/([A-Z])/g, ' $1').replace(/^./, s => s.toUpperCase()).trim();
};

export function ObjectsPanel({ objects, selectedObjectId, onObjectSelect }: ObjectsPanelProps) {
    const movable = objects.filter(o => o.type === 'movable');
    const structural = objects.filter(o => o.type === 'structural');

    return (
        <div className="bg-white rounded-2xl border border-gray-100 p-5 shadow-sm">
            <h3 className="text-sm font-semibold text-gray-700 mb-4 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-[#6b7aa1]"></span>
                Detected Objects ({objects.length})
            </h3>

            {/* Movable Objects */}
            {movable.length > 0 && (
                <div className="mb-4">
                    <div className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-green-400"></span>
                        Movable Furniture ({movable.length})
                    </div>
                    <div className="flex flex-wrap gap-2">
                        {movable.map(obj => {
                            const style = getObjectStyle(obj.label);
                            const isSelected = obj.id === selectedObjectId;
                            const icon = getObjectIcon(obj.label);

                            return (
                                <button
                                    key={obj.id}
                                    onClick={() => onObjectSelect(obj.id)}
                                    className={`
                    inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full
                    border transition-all duration-150 text-sm
                    ${style.bg} ${style.border} ${style.text}
                    ${isSelected
                                            ? 'ring-2 ring-[#6b7aa1] ring-offset-1 shadow-md scale-105'
                                            : 'hover:shadow-sm hover:scale-102'
                                        }
                  `}
                                >
                                    <span className="text-base">{icon}</span>
                                    <span className="font-medium">{formatLabel(obj.label)}</span>
                                    {obj.is_locked && <span className="text-xs">🔒</span>}
                                </button>
                            );
                        })}
                    </div>
                </div>
            )}

            {/* Structural Objects */}
            {structural.length > 0 && (
                <div>
                    <div className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-gray-400"></span>
                        Structural Elements ({structural.length})
                    </div>
                    <div className="flex flex-wrap gap-2">
                        {structural.map(obj => {
                            const icon = getObjectIcon(obj.label);

                            return (
                                <div
                                    key={obj.id}
                                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg
                    bg-gray-50 border border-gray-200 text-gray-500 text-sm"
                                >
                                    <span className="text-base opacity-70">{icon}</span>
                                    <span>{formatLabel(obj.label)}</span>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}
        </div>
    );
}