"use client";

import React, { useRef, useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Eraser, Pencil, Trash2, Maximize2, Minimize2, X } from 'lucide-react';

interface DrawingCanvasProps {
  onDraw: (canvas: HTMLCanvasElement) => void;
  onClear: () => void;
}

export function DrawingCanvas({ onDraw, onClear }: DrawingCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [lastPos, setLastPos] = useState({ x: 0, y: 0 });
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set initial canvas state
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 20; // Thick enough for 28x28 downscaling
  }, []);

  const startDrawing = (e: React.MouseEvent | React.TouchEvent) => {
    if ('touches' in e) {
      // Prevent scrolling on touch
      if (e.cancelable) e.preventDefault();
    }
    const pos = getPos(e);
    setIsDrawing(true);
    setLastPos(pos);
  };

  const draw = (e: React.MouseEvent | React.TouchEvent) => {
    if (!isDrawing || !canvasRef.current) return;

    if ('touches' in e) {
      // Prevent scrolling on touch
      if (e.cancelable) e.preventDefault();
    }

    const pos = getPos(e);
    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;

    ctx.beginPath();
    ctx.moveTo(lastPos.x, lastPos.y);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();

    setLastPos(pos);
    onDraw(canvasRef.current);
  };

  const stopDrawing = () => {
    setIsDrawing(false);
  };

  const getPos = (e: React.MouseEvent | React.TouchEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };

    const rect = canvas.getBoundingClientRect();
    const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
    const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;

    return {
      x: clientX - rect.left,
      y: clientY - rect.top
    };
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    onClear();
  };

  return (
    <>
      <div className="flex flex-col items-center gap-4 p-6 glass rounded-2xl">
        <div className="relative group">
          <canvas
            ref={canvasRef}
            width={280}
            height={280}
            style={{ touchAction: 'none' }}
            className="rounded-lg shadow-inner cursor-crosshair border-2 border-white/5 bg-black"
            onMouseDown={startDrawing}
            onMouseMove={draw}
            onMouseUp={stopDrawing}
            onMouseLeave={stopDrawing}
            onTouchStart={startDrawing}
            onTouchMove={draw}
            onTouchEnd={stopDrawing}
          />
          <div className="absolute top-2 right-2 flex flex-col gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
            <button
              onClick={() => setIsFullscreen(true)}
              className="p-2 bg-primary/20 hover:bg-primary/40 text-primary rounded-lg backdrop-blur-md transition-colors"
              title="Full Screen Mode"
            >
              <Maximize2 size={18} />
            </button>
            <button
              onClick={clearCanvas}
              className="p-2 bg-red-500/20 hover:bg-red-500/40 text-red-500 rounded-lg backdrop-blur-md transition-colors"
              title="Clear Canvas"
            >
              <Trash2 size={18} />
            </button>
          </div>
        </div>

        <div className="flex gap-4 w-full justify-between items-center text-sm font-medium text-slate-400">
          <div className="flex items-center gap-2">
            <Pencil size={14} className="text-primary" />
            <span>Draw a digit (0-9)</span>
          </div>
          <button
            onClick={clearCanvas}
            className="hover:text-primary transition-colors flex items-center gap-1"
          >
            <Eraser size={14} /> Clear
          </button>
        </div>
      </div>

      {/* Fullscreen Overlay */}
      {isFullscreen && (
        <div className="fixed inset-0 z-50 bg-[#0f172a] flex flex-col items-center justify-center p-4">
          <div className="absolute top-6 right-6 flex gap-4">
            <button
              onClick={clearCanvas}
              className="p-3 bg-red-500/20 text-red-500 rounded-full backdrop-blur-xl border border-red-500/20 hover:bg-red-500/40 transition-all"
            >
              <Trash2 size={24} />
            </button>
            <button
              onClick={() => setIsFullscreen(false)}
              className="p-3 bg-white/10 text-white rounded-full backdrop-blur-xl border border-white/20 hover:bg-white/20 transition-all"
            >
              <X size={24} />
            </button>
          </div>

          <div className="w-full max-w-md aspect-square relative">
            <canvas
              ref={canvasRef}
              width={280}
              height={280}
              style={{ touchAction: 'none', width: '100%', height: '100%' }}
              className="rounded-3xl shadow-2xl cursor-crosshair border-4 border-primary/20 bg-black touch-none"
              onMouseDown={startDrawing}
              onMouseMove={draw}
              onMouseUp={stopDrawing}
              onMouseLeave={stopDrawing}
              onTouchStart={startDrawing}
              onTouchMove={draw}
              onTouchEnd={stopDrawing}
            />
          </div>

          <div className="mt-8 text-center space-y-2">
            <h2 className="text-2xl font-bold text-white">Draw your digit</h2>
            <p className="text-slate-400">The model will predict in real-time</p>
          </div>

          <button
            onClick={() => setIsFullscreen(false)}
            className="mt-12 px-8 py-3 bg-primary text-white rounded-full font-bold shadow-lg shadow-primary/20 hover:scale-105 transition-transform"
          >
            Done Drawing
          </button>
        </div>
      )}
    </>
  );
}
