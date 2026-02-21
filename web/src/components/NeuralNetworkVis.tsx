"use client";

import React, { useMemo } from 'react';
import { motion } from 'framer-motion';

/**
 * NeuralWebVis: A premium SVG-based neural network visualization.
 * Features:
 * - Curved synapses for an organic "web" feel.
 * - Signal particles that flow from left to right.
 * - Structured grid layout for "neatness".
 * - Dynamic highlighting based on activations.
 */

interface Point {
    x: number;
    y: number;
}

interface NeuralWebVisProps {
    activations: {
        conv1: number[];
        conv2: number[];
        fc1: number[];
        fc2: number[];
    } | null;
}

export function NeuralNetworkVis({ activations }: NeuralWebVisProps) {
    const width = 800;
    const height = 400;
    const paddingX = 80;
    const paddingY = 60;

    const layerNames = ["Input", "Conv 1", "Conv 2", "FC 1", "Output"];
    // Subset counts for a neat grid
    const layerCounts = [16, 12, 12, 8, 10];

    // 1. Generate Neuron positions
    const layers = useMemo(() => {
        const xStep = (width - paddingX * 2) / (layerCounts.length - 1);
        return layerCounts.map((count, lIdx) => {
            const x = paddingX + lIdx * xStep;
            const yStep = (height - paddingY * 2) / (count - 1);
            return Array.from({ length: count }).map((_, nIdx) => ({
                x,
                y: paddingY + nIdx * yStep,
            }));
        });
    }, [width, height]);

    // 2. Generate Synapses (Curved Paths)
    const synapses = useMemo(() => {
        const paths: { d: string; l1: number; l2: number; n1: number; n2: number }[] = [];
        for (let l = 0; l < layers.length - 1; l++) {
            const currLayer = layers[l];
            const nextLayer = layers[l + 1];

            // Connect each neuron to its counterpart and immediate neighbors in next layer
            currLayer.forEach((p1, i) => {
                const targetIdx = Math.floor((i / currLayer.length) * nextLayer.length);
                const isLastLayer = l === layers.length - 2;
                const range = isLastLayer ? 4 : 2; // Broaden connections to Output layer

                for (let j = targetIdx - range; j <= targetIdx + range; j++) {
                    if (j >= 0 && j < nextLayer.length) {
                        const p2 = nextLayer[j];
                        const cp1x = p1.x + (p2.x - p1.x) / 2;
                        const d = `M ${p1.x} ${p1.y} C ${cp1x} ${p1.y}, ${cp1x} ${p2.y}, ${p2.x} ${p2.y}`;
                        paths.push({ d, l1: l, l2: l + 1, n1: i, n2: j });
                    }
                }
            });
        }
        return paths;
    }, [layers]);

    // Helper to get activation
    const getAct = (lIdx: number, nIdx: number) => {
        if (!activations) return 0;
        const keys = ["conv1", "conv1", "conv2", "fc1", "fc2"];
        const acts = activations[keys[lIdx] as keyof typeof activations] || [];
        // For output layer, index directly
        if (lIdx === 4) return acts[nIdx] || 0;
        return acts[nIdx % acts.length] || 0;
    };

    return (
        <div className="glass rounded-2xl p-4 w-full aspect-[2/1] relative overflow-hidden group border border-white/5 bg-slate-900/40">
            {/* Background Neural Grid */}
            <div className="absolute inset-0 opacity-[0.03] pointer-events-none"
                style={{ backgroundImage: 'radial-gradient(circle, #fff 1px, transparent 1px)', backgroundSize: '20px 20px' }} />

            <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full drop-shadow-2xl">
                <defs>
                    <filter id="glow">
                        <feGaussianBlur stdDeviation="2.5" result="coloredBlur" />
                        <feMerge>
                            <feMergeNode in="coloredBlur" />
                            <feMergeNode in="SourceGraphic" />
                        </feMerge>
                    </filter>

                    {/* Floating particle def */}
                    <radialGradient id="particleGrad">
                        <stop offset="0%" stopColor="#38bdf8" stopOpacity="1" />
                        <stop offset="100%" stopColor="#38bdf8" stopOpacity="0" />
                    </radialGradient>
                </defs>

                {/* Synapses (Paths) */}
                <g>
                    {synapses.map((s, i) => {
                        const a1 = getAct(s.l1, s.n1);
                        const a2 = getAct(s.l2, s.n2);
                        const intensity = (a1 + a2) / 2;
                        const isActive = intensity > 0.35;
                        return (
                            <React.Fragment key={i}>
                                <motion.path
                                    d={s.d}
                                    fill="none"
                                    stroke={isActive ? "var(--primary)" : "rgba(255,255,255,0.03)"}
                                    strokeWidth={isActive ? 1.5 : 0.5}
                                    initial={{ opacity: 0.1 }}
                                    animate={{
                                        opacity: isActive ? 0.4 : 0.05,
                                        strokeWidth: isActive ? 1.5 : 0.5
                                    }}
                                    transition={{ duration: 0.5 }}
                                />
                                {isActive && (
                                    <motion.circle
                                        r={1.5}
                                        fill="#38bdf8"
                                        filter="url(#glow)"
                                    >
                                        <animateMotion
                                            dur={`${1 + Math.random() * 1.5}s`}
                                            repeatCount="infinity"
                                            path={s.d}
                                        />
                                    </motion.circle>
                                )}
                            </React.Fragment>
                        );
                    })}
                </g>

                {/* Neurons */}
                {layers.map((layer, lIdx) => (
                    <g key={lIdx}>
                        {layer.map((p, nIdx) => {
                            const act = getAct(lIdx, nIdx);
                            const isActive = act > 0.4;
                            const isOutput = lIdx === layers.length - 1;

                            return (
                                <g key={nIdx}>
                                    {/* Connection Node */}
                                    <motion.circle
                                        cx={p.x}
                                        cy={p.y}
                                        r={isActive ? (isOutput ? 5 : 4) : 2}
                                        fill={isActive ? "#38bdf8" : "#1e293b"}
                                        stroke="rgba(255,255,255,0.1)"
                                        strokeWidth={0.5}
                                        animate={{
                                            scale: isActive ? [1, 1.2, 1] : 1,
                                            boxShadow: isActive ? "0 0 15px #38bdf8" : "none"
                                        }}
                                        transition={{ repeat: Infinity, duration: 2, delay: nIdx * 0.1 }}
                                        filter={isActive ? "url(#glow)" : ""}
                                    />

                                    {/* Digit Labels for Output Layer */}
                                    {isOutput && (
                                        <text
                                            x={p.x + 15}
                                            y={p.y + 4}
                                            className={`text-[10px] font-bold ${isActive ? 'fill-primary' : 'fill-slate-600'}`}
                                            style={{ filter: isActive ? 'url(#glow)' : '' }}
                                        >
                                            {nIdx}
                                        </text>
                                    )}

                                    {/* Visual pulse for active output */}
                                    {isOutput && isActive && act > 0.8 && (
                                        <motion.circle
                                            cx={p.x}
                                            cy={p.y}
                                            r={10}
                                            fill="none"
                                            stroke="#38bdf8"
                                            strokeWidth={1}
                                            animate={{ scale: [1, 2], opacity: [0.5, 0] }}
                                            transition={{ repeat: Infinity, duration: 1 }}
                                        />
                                    )}
                                </g>
                            );
                        })}

                        {/* Layer Label */}
                        <text
                            x={layer[0].x}
                            y={height - 15}
                            textAnchor="middle"
                            className="fill-slate-500 text-[11px] font-black uppercase tracking-[0.2em] pointer-events-none opacity-50"
                        >
                            {layerNames[lIdx]}
                        </text>
                    </g>
                ))}
            </svg>

            {/* Overall Flow Decorative Overlays */}
            <div className="absolute inset-0 pointer-events-none bg-gradient-to-r from-slate-900/80 via-transparent to-slate-900/60" />
        </div>
    );
}
