'use client';

import { ArrowRight, MoveDown } from 'lucide-react';

interface LandingPageProps {
    onStart: () => void;
}

export function LandingPage({ onStart }: LandingPageProps) {
    return (
        <div className="min-h-screen flex flex-col bg-white text-[#1a1a1a]">
            {/* Navbar */}
            <header className="px-6 py-6 md:px-12 flex justify-between items-center">
                <div className="text-xl font-bold border border-black px-2 py-1">
                    PocketPlanner
                </div>
                <nav className="hidden md:flex gap-8 text-sm font-medium text-gray-500">
                    <a href="#" className="hover:text-black transition-colors">Home</a>
                    <a href="#" className="hover:text-black transition-colors">Services</a>
                    <a href="#" className="hover:text-black transition-colors">Contact</a>
                    <a href="#" className="hover:text-black transition-colors">Support</a>
                </nav>
                <button
                    onClick={onStart}
                    className="bg-black text-white px-6 py-2 text-sm font-medium hover:bg-gray-800 transition-colors"
                >
                    Sign Up
                </button>
            </header>

            {/* Hero Section */}
            <main className="flex-1 px-6 md:px-12 py-12 md:py-20 max-w-7xl mx-auto w-full">
                <div className="grid md:grid-cols-2 gap-12 items-center">

                    {/* Left Content */}
                    <div className="space-y-8">
                        <h1 className="text-6xl md:text-8xl font-normal tracking-tight leading-[0.9]">
                            Interior Design
                        </h1>

                        <p className="text-gray-500 max-w-md text-lg leading-relaxed">
                            Step into a world where the art of Interior Design is meticulously
                            crafted to bring together timeless elegance and cutting-edge
                            modern Innovation.
                        </p>

                        <button
                            onClick={onStart}
                            className="bg-[#1a1a1a] text-white px-8 py-4 text-sm font-medium hover:bg-black transition-all flex items-center gap-2 group"
                        >
                            Start Project
                            <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                        </button>

                        {/* Stats */}
                        <div className="flex gap-12 pt-12">
                            <div>
                                <div className="text-4xl font-light text-gray-800">400+</div>
                                <div className="text-sm text-gray-500 mt-1">Project Complete</div>
                            </div>
                            <div>
                                <div className="text-4xl font-light text-gray-800">600+</div>
                                <div className="text-sm text-gray-500 mt-1">Satisfied Clients</div>
                            </div>
                            <div>
                                <div className="text-4xl font-light text-gray-800">100+</div>
                                <div className="text-sm text-gray-500 mt-1">Unique Styles</div>
                            </div>
                        </div>
                    </div>

                    {/* Right Content - Images */}
                    <div className="relative h-[600px] hidden md:block">
                        {/* Main large image */}
                        <div className="absolute top-0 right-0 w-[80%] h-[60%] bg-[#f5f5f5] overflow-hidden">
                            <img
                                src="https://images.unsplash.com/photo-1616486338812-3dadae4b4ace?auto=format&fit=crop&q=80&w=1000"
                                alt="Minimal Interior"
                                className="w-full h-full object-cover"
                            />
                        </div>

                        {/* Overlapping bottom image */}
                        <div className="absolute bottom-0 left-0 w-[70%] h-[50%] bg-[#eaeaea] overflow-hidden border-8 border-white">
                            <img
                                src="https://images.unsplash.com/photo-1598928506311-c55ded91a20c?auto=format&fit=crop&q=80&w=1000"
                                alt="Living Room"
                                className="w-full h-full object-cover"
                            />
                        </div>

                        {/* Decorative arrow box */}
                        <div className="absolute bottom-12 right-12 w-24 h-24 bg-[#1a1a1a] flex items-center justify-center">
                            <MoveDown className="text-white w-8 h-8 animate-bounce" />
                        </div>
                    </div>
                </div>

                {/* Services Section */}
                <div className="mt-32 border-t pt-12 border-gray-100">
                    <div className="flex items-center gap-4 mb-12">
                        <div className="h-px bg-black w-12"></div>
                        <h3 className="text-2xl font-medium">Our Services</h3>
                    </div>

                    <div className="grid md:grid-cols-3 gap-12">
                        <div className="space-y-4 cursor-pointer group">
                            <div className="w-12 h-12 flex items-center justify-center border border-gray-200 rounded-full group-hover:border-black transition-colors">
                                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
                            </div>
                            <h4 className="font-medium text-lg">Lighting Design</h4>
                            <p className="text-sm text-gray-500 leading-relaxed">Achieve the perfect balance of ambient, task, and accent lighting.</p>
                        </div>

                        <div className="space-y-4 cursor-pointer group">
                            <div className="w-12 h-12 flex items-center justify-center border border-gray-200 rounded-full group-hover:border-black transition-colors">
                                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" /></svg>
                            </div>
                            <h4 className="font-medium text-lg">Interior Design</h4>
                            <p className="text-sm text-gray-500 leading-relaxed">From concept to completion, we craft spaces that reflect your style.</p>
                        </div>

                        <div className="space-y-4 cursor-pointer group">
                            <div className="w-12 h-12 flex items-center justify-center border border-gray-200 rounded-full group-hover:border-black transition-colors">
                                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" /></svg>
                            </div>
                            <h4 className="font-medium text-lg">Outdoor Design</h4>
                            <p className="text-sm text-gray-500 leading-relaxed">Celebrate the changing seasons with tailored outdoor sanctuaries.</p>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
}
