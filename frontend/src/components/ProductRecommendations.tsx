'use client';

import { useState } from 'react';
import { ArrowLeft, Loader2, ShoppingBag, ExternalLink, DollarSign, Star, Search } from 'lucide-react';
import type { ShopResponse, RoomObject, RoomDimensions } from '@/lib/types';

interface ProductRecommendationsProps {
    shopData: ShopResponse | null;
    isLoading: boolean;
    budget: number;
    onBudgetChange: (budget: number) => void;
    onSearch: () => void;
    onBack: () => void;
    hasLayout: boolean;
}

export function ProductRecommendations({
    shopData,
    isLoading,
    budget,
    onBudgetChange,
    onSearch,
    onBack,
    hasLayout,
}: ProductRecommendationsProps) {

    if (isLoading) {
        return (
            <div className="flex flex-col items-center justify-center min-h-[500px] gap-4">
                <Loader2 className="w-12 h-12 animate-spin text-black" />
                <p className="text-black text-lg font-medium">Finding products for your room...</p>
                <p className="text-gray-400 text-sm">Searching Google Shopping for matching furniture</p>
            </div>
        );
    }

    return (
        <div className="space-y-8 animate-in fade-in duration-500">
            {/* Header */}
            <div className="flex items-center justify-between border-b border-gray-100 pb-6">
                <div className="flex items-center gap-4">
                    <button
                        onClick={onBack}
                        className="p-2 bg-black text-white rounded-full hover:bg-gray-800 transition-colors"
                    >
                        <ArrowLeft className="w-5 h-5" />
                    </button>
                    <div>
                        <h2 className="text-2xl font-bold text-black tracking-tight">Shop Your Room</h2>
                        <p className="text-sm text-gray-400">Find real products matching your design</p>
                    </div>
                </div>
            </div>

            {/* Budget Input + Search */}
            <div className="flex flex-col sm:flex-row items-end gap-4 p-6 border border-gray-100 bg-gray-50/50">
                <div className="flex-1 w-full">
                    <label className="block text-xs font-semibold text-gray-500 uppercase tracking-widest mb-2">
                        Total Room Budget
                    </label>
                    <div className="relative">
                        <DollarSign className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                        <input
                            type="number"
                            value={budget}
                            onChange={(e) => onBudgetChange(Number(e.target.value))}
                            className="w-full pl-9 pr-4 py-3 border border-gray-200 bg-white text-lg font-medium focus:outline-none focus:border-black transition-colors rounded-full"
                            placeholder="2000"
                        />
                    </div>
                </div>
                <button
                    onClick={onSearch}
                    disabled={!hasLayout || isLoading}
                    className="w-full sm:w-auto px-8 py-3 bg-black text-white font-medium hover:bg-gray-800 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 rounded-full"
                >
                    <Search className="w-4 h-4" />
                    Find Products
                </button>
            </div>

            {/* Summary */}
            {shopData && (
                <div className="flex gap-6 text-sm">
                    <div className="border border-gray-100 px-5 py-3">
                        <span className="text-gray-400">Budget:</span>{' '}
                        <span className="font-semibold text-black">${shopData.total_budget.toLocaleString()}</span>
                    </div>
                    <div className="border border-gray-100 px-5 py-3">
                        <span className="text-gray-400">Estimated Total:</span>{' '}
                        <span className={`font-semibold ${shopData.total_estimated <= shopData.total_budget ? 'text-green-600' : 'text-red-500'}`}>
                            ${shopData.total_estimated.toLocaleString()}
                        </span>
                    </div>
                    <div className="border border-gray-100 px-5 py-3">
                        <span className="text-gray-400">Items:</span>{' '}
                        <span className="font-semibold text-black">{shopData.items.length}</span>
                    </div>
                </div>
            )}

            {/* Product Cards by Furniture Item */}
            {shopData && shopData.items.map((item) => (
                <div key={item.furniture_id} className="border border-gray-100">
                    {/* Item Header */}
                    <div className="flex items-center justify-between p-4 border-b border-gray-50 bg-gray-50/30">
                        <div>
                            <h3 className="font-semibold text-black capitalize">
                                {item.furniture_label.replace(/_/g, ' ')}
                            </h3>
                            <p className="text-xs text-gray-400 mt-0.5">
                                Budget: ${item.budget_allocated.toFixed(0)} · Query: "{item.search_query}"
                            </p>
                        </div>
                        {item.error && (
                            <span className="text-xs text-red-400">Search failed</span>
                        )}
                    </div>

                    {/* Products Grid */}
                    {item.products.length > 0 ? (
                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 p-4">
                            {item.products.map((product, idx) => (
                                <a
                                    key={idx}
                                    href={product.link}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="group border border-gray-100 hover:border-black transition-all p-3 flex flex-col gap-3"
                                >
                                    {/* Thumbnail */}
                                    <div className="aspect-square bg-gray-50 overflow-hidden flex items-center justify-center">
                                        {product.thumbnail ? (
                                            <img
                                                src={product.thumbnail}
                                                alt={product.title}
                                                className="w-full h-full object-contain mix-blend-multiply group-hover:scale-105 transition-transform"
                                            />
                                        ) : (
                                            <ShoppingBag className="w-8 h-8 text-gray-300" />
                                        )}
                                    </div>

                                    {/* Info */}
                                    <div className="flex-1 flex flex-col gap-1">
                                        <p className="text-sm font-medium text-gray-800 line-clamp-2 leading-snug">
                                            {product.title}
                                        </p>
                                        <p className="text-xs text-gray-400">{product.source}</p>
                                    </div>

                                    {/* Price + Rating */}
                                    <div className="flex items-center justify-between pt-2 border-t border-gray-50">
                                        <span className="text-lg font-bold text-black">
                                            {product.price_raw || (product.price ? `$${product.price}` : 'N/A')}
                                        </span>
                                        <div className="flex items-center gap-1">
                                            {product.rating && (
                                                <>
                                                    <Star className="w-3 h-3 text-amber-400 fill-amber-400" />
                                                    <span className="text-xs text-gray-500">{product.rating}</span>
                                                </>
                                            )}
                                            <ExternalLink className="w-3 h-3 text-gray-300 group-hover:text-black transition-colors ml-1" />
                                        </div>
                                    </div>
                                </a>
                            ))}
                        </div>
                    ) : (
                        <div className="p-6 text-center text-sm text-gray-400">
                            No products found within budget. Try increasing your budget.
                        </div>
                    )}
                </div>
            ))}

            {/* Empty state */}
            {!shopData && (
                <div className="text-center py-16 text-gray-400">
                    <ShoppingBag className="w-12 h-12 mx-auto mb-4 opacity-30" />
                    <p className="text-lg">Set your budget and click "Find Products"</p>
                    <p className="text-sm mt-1">We'll search Google Shopping for furniture matching your room</p>
                </div>
            )}
        </div>
    );
}