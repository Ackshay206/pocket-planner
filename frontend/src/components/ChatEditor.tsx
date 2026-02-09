'use client';

import { useState, useRef, useEffect } from 'react';
import { ArrowLeft, Send, Loader2, Sparkles } from 'lucide-react';
import type { RoomObject, RoomDimensions } from '@/lib/types';

interface ChatMessage {
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
}

interface ChatEditorProps {
    imageBase64: string | null;
    layout: RoomObject[];
    roomDimensions: RoomDimensions | null;
    messages: ChatMessage[];
    isLoading: boolean;
    onSendCommand: (command: string) => void;
    onBack: () => void;
}

export function ChatEditor({
    imageBase64,
    layout,
    roomDimensions,
    messages,
    isLoading,
    onSendCommand,
    onBack,
}: ChatEditorProps) {
    const [input, setInput] = useState('');
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    // Auto-scroll to bottom when messages change
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;

        onSendCommand(input.trim());
        setInput('');
    };

    const suggestedCommands = [
        'Move the bed to the left',
        'Rotate the desk 90 degrees',
        'Remove the nightstand',
        'Add a plant in the corner',
    ];

    return (
        <div className="flex flex-col lg:flex-row gap-6 h-[calc(100vh-200px)] min-h-[500px]">
            {/* Image Section */}
            <div className="flex-1 flex flex-col">
                <div className="flex items-center gap-4 mb-4">
                    <button
                        onClick={onBack}
                        className="p-2 rounded-full hover:bg-gray-100 transition-colors"
                    >
                        <ArrowLeft className="w-5 h-5 text-gray-600" />
                    </button>
                    <div>
                        <h2 className="text-xl font-semibold text-gray-800">Chat Editor</h2>
                        <p className="text-sm text-gray-500">Use natural language to edit your room</p>
                    </div>
                </div>

                <div className="flex-1 floor-plan-container overflow-hidden">
                    {imageBase64 ? (
                        <img
                            src={`data:image/png;base64,${imageBase64}`}
                            alt="Room view"
                            className="w-full h-full object-contain rounded-xl"
                        />
                    ) : (
                        <div className="h-full bg-gradient-to-br from-[#f5f3f0] to-[#e8e6e3] rounded-xl flex items-center justify-center">
                            <p className="text-gray-400">No image available</p>
                        </div>
                    )}
                </div>
            </div>

            {/* Chat Section */}
            <div className="w-full lg:w-96 flex flex-col bg-white rounded-2xl shadow-sm border border-gray-100">
                {/* Chat Header */}
                <div className="p-4 border-b border-gray-100">
                    <div className="flex items-center gap-2">
                        <Sparkles className="w-5 h-5 text-[#6b7aa1]" />
                        <h3 className="font-semibold text-gray-800">AI Assistant</h3>
                    </div>
                    <p className="text-xs text-gray-400 mt-1">
                        Describe changes in natural language
                    </p>
                </div>

                {/* Messages */}
                <div className="flex-1 overflow-y-auto p-4 space-y-4">
                    {messages.length === 0 ? (
                        <div className="space-y-4">
                            <p className="text-sm text-gray-500 text-center">
                                Try some suggestions:
                            </p>
                            {suggestedCommands.map((cmd, i) => (
                                <button
                                    key={i}
                                    onClick={() => onSendCommand(cmd)}
                                    disabled={isLoading}
                                    className="w-full p-3 text-left text-sm bg-black text-white hover:bg-gray-800 rounded-full transition-colors disabled:opacity-50"
                                >
                                    "{cmd}"
                                </button>
                            ))}
                        </div>
                    ) : (
                        messages.map((msg, i) => (
                            <div
                                key={i}
                                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                            >
                                <div
                                    className={`max-w-[80%] p-3 rounded-[2rem] ${msg.role === 'user'
                                        ? 'bg-black text-white'
                                        : 'bg-[#f5f3f0] text-gray-800'
                                        }`}
                                >
                                    <p className="text-sm">{msg.content}</p>
                                    <p className={`text-xs mt-1 ${msg.role === 'user' ? 'text-white/60' : 'text-gray-400'
                                        }`}>
                                        {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                    </p>
                                </div>
                            </div>
                        ))
                    )}
                    <div ref={messagesEndRef} />
                </div>

                {/* Input */}
                <form onSubmit={handleSubmit} className="p-4 border-t border-gray-100">
                    <div className="flex gap-2">
                        <input
                            ref={inputRef}
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Type your edit command..."
                            disabled={isLoading}
                            className="flex-1 px-4 py-2 rounded-full bg-[#f5f3f0] border-none focus:outline-none focus:ring-2 focus:ring-black disabled:opacity-50"
                        />
                        <button
                            type="submit"
                            disabled={!input.trim() || isLoading}
                            className="p-2 bg-black text-white rounded-full hover:bg-gray-800 transition-colors disabled:opacity-50 disabled:hover:bg-black"
                        >
                            {isLoading ? (
                                <Loader2 className="w-5 h-5 animate-spin" />
                            ) : (
                                <Send className="w-5 h-5" />
                            )}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
}
