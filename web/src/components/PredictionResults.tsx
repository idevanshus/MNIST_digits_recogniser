"use client";

import React from 'react';
import { motion } from 'framer-motion';

interface PredictionResultsProps {
    predictions: number[] | null;
    loading: boolean;
}

export function PredictionResults({ predictions, loading }: PredictionResultsProps) {
    if (loading) {
        return (
            <div className="w-full flex justify-center py-10 animate-pulse text-slate-400 font-medium">
                Neural network is thinking...
            </div>
        );
    }

    if (!predictions) {
        return (
            <div className="w-full text-center py-10 text-slate-500 italic">
                Draw a digit to see the prediction
            </div>
        );
    }

    const maxProb = Math.max(...predictions);
    const isUnknown = maxProb < 0.01; // My threshold logic in page.tsx sets it to 0
    const result = isUnknown ? "?" : predictions.indexOf(maxProb);
    const confidence = isUnknown ? "0.0" : (maxProb * 100).toFixed(1);

    return (
        <div className="w-full space-y-6">
            <div className={`flex items-center justify-between p-6 glass rounded-2xl border-primary/20 ${isUnknown ? 'bg-slate-800' : 'bg-primary/5'}`}>
                <div>
                    <p className="text-slate-400 text-sm font-medium mb-1">Detected Digit</p>
                    <h2 className="text-6xl font-black text-white">{result === "?" ? "Unknown" : result}</h2>
                </div>
                <div className="text-right">
                    <p className="text-slate-400 text-sm font-medium mb-1">Confidence</p>
                    <div className="flex items-center gap-2">
                        <span className={`text-3xl font-bold ${isUnknown ? 'text-slate-500' : 'text-primary'}`}>{confidence}%</span>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-5 gap-3">
                {predictions.map((prob, i) => (
                    <div key={i} className="flex flex-col items-center">
                        <div className="w-full h-24 bg-slate-800/50 rounded-lg relative overflow-hidden">
                            <motion.div
                                initial={{ height: 0 }}
                                animate={{ height: `${prob * 100}%` }}
                                className={`absolute bottom-0 w-full transition-colors ${i === result ? 'bg-primary' : 'bg-slate-600'}`}
                            />
                        </div>
                        <span className={`mt-2 font-bold ${i === result ? 'text-primary' : 'text-slate-500'}`}>{i}</span>
                    </div>
                ))}
            </div>
        </div>
    );
}
