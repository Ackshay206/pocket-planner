'use client';

import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Image as ImageIcon, X } from 'lucide-react';

interface ImageUploadProps {
    onImageSelect: (base64: string) => void;
    currentImage?: string | null;
    disabled?: boolean;
}

export function ImageUpload({
    onImageSelect,
    currentImage,
    disabled = false,
}: ImageUploadProps) {
    const [preview, setPreview] = useState<string | null>(currentImage || null);

    const onDrop = useCallback(
        (acceptedFiles: File[]) => {
            const file = acceptedFiles[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = () => {
                const result = reader.result as string;
                setPreview(result);
                // Extract base64 part (remove data:image/...;base64, prefix)
                const base64 = result.split(',')[1];
                onImageSelect(base64);
            };
            reader.readAsDataURL(file);
        },
        [onImageSelect]
    );

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'image/jpeg': ['.jpg', '.jpeg'],
            'image/png': ['.png'],
            'image/webp': ['.webp'],
        },
        maxSize: 10 * 1024 * 1024, // 10MB
        disabled,
        multiple: false,
    });

    const clearImage = (e: React.MouseEvent) => {
        e.stopPropagation();
        setPreview(null);
    };

    return (
        <div
            {...getRootProps()}
            className={`
        relative min-h-[400px] border border-dashed transition-all duration-300 cursor-pointer flex items-center justify-center
        ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
        ${isDragActive
                    ? 'border-black bg-gray-50'
                    : preview
                        ? 'border-gray-200 bg-white'
                        : 'border-gray-300 hover:border-black bg-white hover:bg-gray-50'
                }
      `}
        >
            <input {...getInputProps()} />

            {preview ? (
                <div className="relative w-full h-full min-h-[400px] group flex items-center justify-center bg-gray-100">
                    <img
                        src={preview}
                        alt="Floor plan preview"
                        className="max-w-full max-h-[400px] object-contain p-4"
                    />
                    {!disabled && (
                        <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                            <div className="text-center">
                                <ImageIcon className="w-8 h-8 mx-auto mb-2 text-white" />
                                <p className="text-white font-medium text-sm tracking-wide">CHANGE IMAGE</p>
                            </div>
                        </div>
                    )}
                    {!disabled && (
                        <button
                            onClick={clearImage}
                            className="absolute top-4 right-4 p-2 bg-black text-white hover:bg-gray-800 transition-colors z-10"
                        >
                            <X className="w-4 h-4" />
                        </button>
                    )}
                </div>
            ) : (
                <div className="flex flex-col items-center justify-center h-full p-8 text-center">
                    <div className={`p-4 mb-4 border border-gray-200 rounded-full ${isDragActive ? 'bg-black text-white' : 'bg-white text-gray-400'}`}>
                        <Upload className="w-6 h-6" />
                    </div>
                    <p className="text-lg font-medium text-gray-900 mb-2 tracking-tight">
                        {isDragActive ? 'Drop your floor plan here' : 'Upload Floor Plan'}
                    </p>
                    <p className="text-sm text-gray-400 max-w-xs">
                        Drag & drop or click to browse (JPEG, PNG, WebP • Max 10MB)
                    </p>
                </div>
            )}
        </div>
    );
}
