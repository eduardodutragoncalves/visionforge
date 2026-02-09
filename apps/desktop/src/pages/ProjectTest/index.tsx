import { useState, useRef, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { Card } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { Select } from '../../components/ui/Select';
import { Upload, Camera, Zap } from 'lucide-react';
import { ENGINE_HTTP_URL } from '../../lib/engine';

interface Prediction {
    box?: number[]; // x1, y1, x2, y2
    confidence: number;
    class_name: string;
    class_id: number;
}

interface InferenceResult {
    task?: 'detect' | 'classify';
    predictions: Prediction[];
    latency_ms: number;
    fps_est: number;
}

interface HistoryVariantResult {
    variant_id: string;
    tested_at?: string;
    task?: 'detect' | 'classify';
    predictions: Prediction[];
    latency_ms?: number;
    fps_est?: number;
    feedback?: 'correct' | 'incorrect' | null;
}

interface HistoryItem {
    image_id: string;
    image_file?: string;
    original_name?: string;
    created_at?: string;
    last_tested_at?: string;
    variants?: Record<string, HistoryVariantResult>;
}

export const ProjectTest = () => {
    const { projectId } = useParams();

    // State
    const [selectedVariant, setSelectedVariant] = useState<string>("");
    const [variants, setVariants] = useState<{ id: string, name: string }[]>([]);
    const [imageSrc, setImageSrc] = useState<string | null>(null);
    const [result, setResult] = useState<InferenceResult | null>(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [history, setHistory] = useState<HistoryItem[]>([]);
    const [historyLoading, setHistoryLoading] = useState(false);
    const [activeHistoryId, setActiveHistoryId] = useState<string | null>(null);

    // Refs for Canvas drawing
    const imgRef = useRef<HTMLImageElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);

    // Load Variants on mount
    useEffect(() => {
        const fetchVariants = async () => {
            try {
                const res = await fetch(`${ENGINE_HTTP_URL}/projects/${projectId}/variants`);
                if (res.ok) {
                    const data = await res.json();
                    const normalized = Array.isArray(data) ? data : [];
                    setVariants(normalized);
                    if (normalized.length > 0) setSelectedVariant(normalized[0].id);
                }
            } catch (e) {
                console.error(e);
            }
        };
        fetchVariants();
    }, [projectId]);

    const loadHistory = async () => {
        if (!projectId) return;
        setHistoryLoading(true);
        try {
            const res = await fetch(`${ENGINE_HTTP_URL}/projects/${projectId}/inference/history`);
            if (res.ok) {
                const data = await res.json();
                setHistory(Array.isArray(data) ? data : []);
            }
        } catch (e) {
            console.error(e);
        } finally {
            setHistoryLoading(false);
        }
    };

    useEffect(() => {
        loadHistory();
    }, [projectId]);

    const historyImageUrl = (item: HistoryItem) => {
        if (!projectId) return '';
        return `${ENGINE_HTTP_URL}/projects/${projectId}/inference/history/images/${item.image_id}`;
    };

    const activeHistory = history.find((item) => item.image_id === activeHistoryId) || null;

    const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            const url = URL.createObjectURL(file);
            setImageSrc(url);
            setResult(null); // Clear previous results
            setActiveHistoryId(null);

            // Auto-trigger inference if model is selected? No, wait for user or maybe yes.
            // Let's do it automatically for better UX
            runInference(file);
        }
    };

    const runInference = async (file: File) => {
        if (!selectedVariant) return;

        setIsProcessing(true);
        const formData = new FormData();
        formData.append('file', file);
        formData.append('project_id', projectId || "");

        try {
            const res = await fetch(`${ENGINE_HTTP_URL}/variants/${selectedVariant}/infer`, {
                method: 'POST',
                body: formData
            });

            if (res.ok) {
                const data = await res.json();
                setResult(data);
                if (data.history_item) {
                    setHistory((prev) => {
                        const next = prev.filter((item) => item.image_id !== data.history_item.image_id);
                        return [data.history_item, ...next];
                    });
                    setActiveHistoryId(data.history_item.image_id);
                }
            } else {
                console.error("Inference failed");
            }
        } catch (e) {
            console.error(e);
        } finally {
            setIsProcessing(false);
        }
    };

    const runInferenceForHistory = async (imageId: string, variantId?: string) => {
        const targetVariant = variantId || selectedVariant;
        if (!targetVariant || !projectId) return;
        setIsProcessing(true);
        const formData = new FormData();
        formData.append('conf_threshold', '0.25');
        try {
            const res = await fetch(`${ENGINE_HTTP_URL}/projects/${projectId}/variants/${targetVariant}/infer/history/${imageId}`, {
                method: 'POST',
                body: formData
            });
            if (res.ok) {
                const data = await res.json();
                setResult(data);
                if (data.history_item) {
                    setHistory((prev) => {
                        const next = prev.filter((item) => item.image_id !== data.history_item.image_id);
                        return [data.history_item, ...next];
                    });
                    setActiveHistoryId(data.history_item.image_id);
                }
            } else {
                console.error("Inference failed");
            }
        } catch (e) {
            console.error(e);
        } finally {
            setIsProcessing(false);
        }
    };

    const handleFeedback = async (imageId: string, variantId: string, verdict: 'correct' | 'incorrect' | null) => {
        if (!projectId) return;
        const formData = new FormData();
        formData.append('variant_id', variantId);
        if (verdict) {
            formData.append('verdict', verdict);
        }
        try {
            const res = await fetch(`${ENGINE_HTTP_URL}/projects/${projectId}/inference/history/${imageId}/feedback`, {
                method: 'POST',
                body: formData
            });
            if (!res.ok) return;
            setHistory((prev) =>
                prev.map((item) => {
                    if (item.image_id !== imageId) return item;
                    const variantsMap = { ...(item.variants || {}) };
                    if (variantsMap[variantId]) {
                        variantsMap[variantId] = { ...variantsMap[variantId], feedback: verdict };
                    }
                    return { ...item, variants: variantsMap };
                })
            );
        } catch (e) {
            console.error(e);
        }
    };

    // Draw Overlay
    useEffect(() => {
        if (imageSrc && result && imgRef.current && canvasRef.current) {
            if (result.task === 'classify') {
                const img = imgRef.current;
                const canvas = canvasRef.current;
                if (img && canvas) {
                    canvas.width = img.width;
                    canvas.height = img.height;
                }
                const ctx = canvasRef.current.getContext('2d');
                if (ctx) {
                    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
                }
                return;
            }
            const img = imgRef.current;
            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d');

            if (!ctx) return;

            // Match canvas size to displayed image size
            canvas.width = img.width;
            canvas.height = img.height;

            // Clear
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Calculate scaling factors (Displayed vs Natural)
            const scaleX = img.width / img.naturalWidth;
            const scaleY = img.height / img.naturalHeight;

            // Draw boxes
            result.predictions.forEach(p => {
                if (!p.box) return;
                const [x1, y1, x2, y2] = p.box;

                // Scale coordinates
                const sx1 = x1 * scaleX;
                const sy1 = y1 * scaleY;
                const sw = (x2 - x1) * scaleX;
                const sh = (y2 - y1) * scaleY;

                // Box
                ctx.strokeStyle = '#00f7ff'; // Cyan accent
                ctx.lineWidth = 2;
                ctx.strokeRect(sx1, sy1, sw, sh);

                // Label Background
                ctx.fillStyle = '#00f7ff';
                const label = `${p.class_name} ${(p.confidence * 100).toFixed(1)}%`;
                const textWidth = ctx.measureText(label).width;
                ctx.fillRect(sx1, sy1 - 20, textWidth + 10, 20);

                // Label Text
                ctx.fillStyle = '#000000';
                ctx.font = '12px Inter, sans-serif';
                ctx.fillText(label, sx1 + 5, sy1 - 5);
            });
        }
    }, [result, imageSrc]);

    useEffect(() => {
        if (!activeHistory || !projectId) return;
        const variantResult = activeHistory.variants?.[selectedVariant || ""];
        if (variantResult) {
            setResult({
                task: variantResult.task,
                predictions: variantResult.predictions || [],
                latency_ms: variantResult.latency_ms ?? 0,
                fps_est: variantResult.fps_est ?? 0
            });
        } else {
            setResult(null);
        }
        setImageSrc(historyImageUrl(activeHistory));
    }, [activeHistoryId, selectedVariant, projectId]);

    const renderPredictionSummary = (variantResult: HistoryVariantResult) => {
        if (variantResult.task === 'classify') {
            const top = variantResult.predictions?.[0];
            if (!top) return 'No predictions';
            return `${top.class_name} ${(top.confidence * 100).toFixed(1)}%`;
        }
        const count = variantResult.predictions?.length || 0;
        return `${count} detections`;
    };

    return (
        <div className="project-test animate-fade-in space-y-6 h-full flex flex-col">
            <header className="flex justify-between items-center">
                <div>
                    <h2 className="text-2xl font-bold">Model Playground</h2>
                    <p className="text-secondary">Test your trained models with local images</p>
                </div>
            </header>

            <div className="grid grid-cols-12 gap-6 flex-1">
                <div className="col-span-3 space-y-6">
                {/* Control Panel */}
                <Card className="h-fit space-y-6">
                    <div>
                        <Select
                            label="Select Model"
                            options={variants.map(v => ({ value: v.id, label: v.name }))}
                            value={selectedVariant}
                            onChange={e => setSelectedVariant(e.target.value)}
                        />
                        {variants.length === 0 && (
                            <div className="mt-2 text-xs text-secondary">No trained variants yet.</div>
                        )}
                    </div>

                    <div className="pt-4 border-t border-white/10">
                        <div className="border-2 border-dashed border-white/20 rounded-lg p-6 text-center hover:border-accent-primary/50 transition-colors cursor-pointer relative">
                            <input
                                type="file"
                                accept="image/*"
                                className="absolute inset-0 opacity-0 cursor-pointer"
                                onChange={handleImageUpload}
                            />
                            <Upload className="mx-auto mb-2 text-secondary" />
                            <span className="text-sm text-secondary">Click to Upload Image</span>
                        </div>
                    </div>

                    {result && (
                        <div className="space-y-4 pt-4 border-t border-white/10">
                            <h3 className="font-bold flex items-center gap-2">
                                <Zap size={18} className="text-yellow-400" />
                                Performance
                            </h3>
                            <div className="grid grid-cols-2 gap-4">
                                <div className="bg-white/5 p-3 rounded">
                                    <div className="text-xs text-secondary">Latency</div>
                                    <div className="text-xl font-mono">{result.latency_ms.toFixed(1)}ms</div>
                                </div>
                                <div className="bg-white/5 p-3 rounded">
                                    <div className="text-xs text-secondary">Est. FPS</div>
                                    <div className="text-xl font-mono">{result.fps_est.toFixed(0)}</div>
                                </div>
                            </div>
                            {result.task === 'classify' ? (
                                <div className="bg-white/5 p-3 rounded">
                                    <div className="text-xs text-secondary mb-2">Top Predictions</div>
                                    <div className="space-y-2">
                                        {result.predictions.map((p, i) => (
                                            <div key={i} className="flex items-center justify-between text-xs">
                                                <span className="text-secondary">{p.class_name}</span>
                                                <span className="text-bone font-mono">{(p.confidence * 100).toFixed(1)}%</span>
                                            </div>
                                        ))}
                                        {result.predictions.length === 0 && <span className="text-xs text-secondary">None</span>}
                                    </div>
                                </div>
                            ) : (
                                <div className="bg-white/5 p-3 rounded">
                                    <div className="text-xs text-secondary mb-2">Detections</div>
                                    <div className="flex flex-wrap gap-2">
                                        {result.predictions.map((p, i) => (
                                            <span key={i} className="text-xs px-2 py-1 rounded bg-accent-primary/20 text-accent-primary">
                                                {p.class_name}
                                            </span>
                                        ))}
                                        {result.predictions.length === 0 && <span className="text-xs text-secondary">None</span>}
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </Card>

                <Card className="h-fit space-y-4">
                    <div className="flex items-center justify-between">
                        <h3 className="text-lg font-semibold">History</h3>
                        <Button size="sm" variant="secondary" onClick={loadHistory} disabled={historyLoading}>
                            {historyLoading ? 'Loading...' : 'Refresh'}
                        </Button>
                    </div>

                    {!historyLoading && history.length === 0 && (
                        <div className="text-sm text-secondary">No history yet. Upload an image to start.</div>
                    )}

                    {history.length > 0 && (
                        <div className="grid grid-cols-3 gap-2 max-h-48 overflow-y-auto">
                            {history.map((item) => (
                                <button
                                    key={item.image_id}
                                    type="button"
                                    onClick={() => setActiveHistoryId(item.image_id)}
                                    className={`border-2 ${activeHistoryId === item.image_id ? 'border-accent-primary' : 'border-graphite-700'} bg-graphite-900 p-1`}
                                >
                                    <img
                                        src={historyImageUrl(item)}
                                        alt={item.original_name || item.image_id}
                                        className="h-16 w-full object-cover"
                                    />
                                </button>
                            ))}
                        </div>
                    )}

                    {activeHistory && (
                        <div className="space-y-3">
                            <div className="text-xs uppercase tracking-[0.3em] text-secondary">
                                Comparisons
                            </div>
                            {variants.map((variant) => {
                                const variantResult = activeHistory.variants?.[variant.id];
                                const feedback = variantResult?.feedback || null;
                                return (
                                    <div key={variant.id} className="border border-graphite-700/70 bg-graphite-900 p-3">
                                        <div className="flex items-center justify-between">
                                            <div className="text-sm font-semibold">{variant.name}</div>
                                            {variantResult ? (
                                                <span className="text-xs text-secondary">{renderPredictionSummary(variantResult)}</span>
                                            ) : (
                                                <Button
                                                    size="sm"
                                                    variant="ghost"
                                                    onClick={() => runInferenceForHistory(activeHistory.image_id, variant.id)}
                                                >
                                                    Run
                                                </Button>
                                            )}
                                        </div>
                                        {variantResult && (
                                            <div className="mt-3 flex items-center gap-2">
                                                <Button
                                                    size="sm"
                                                    variant="secondary"
                                                    className={feedback === 'correct' ? 'border-accent-success text-accent-success' : ''}
                                                    onClick={() => handleFeedback(activeHistory.image_id, variant.id, feedback === 'correct' ? null : 'correct')}
                                                >
                                                    Correct
                                                </Button>
                                                <Button
                                                    size="sm"
                                                    variant="secondary"
                                                    className={feedback === 'incorrect' ? 'border-accent-error text-accent-error' : ''}
                                                    onClick={() => handleFeedback(activeHistory.image_id, variant.id, feedback === 'incorrect' ? null : 'incorrect')}
                                                >
                                                    Wrong
                                                </Button>
                                                <span className="text-xs text-secondary">
                                                    {variantResult.tested_at ? new Date(variantResult.tested_at).toLocaleTimeString() : ''}
                                                </span>
                                            </div>
                                        )}
                                    </div>
                                );
                            })}
                            {variants.length === 0 && (
                                <div className="text-sm text-secondary">No variants available yet.</div>
                            )}
                        </div>
                    )}
                </Card>
                </div>

                {/* Visualizer */}
                <div className="col-span-9 bg-black/40 rounded-lg border border-white/5 flex items-center justify-center p-4 relative overflow-hidden min-h-[400px]">
                    {!imageSrc && (
                        <div className="text-center text-secondary">
                            <Camera size={48} className="mx-auto mb-4 opacity-50" />
                            <p>Upload an image to start inference</p>
                        </div>
                    )}

                    {imageSrc && (
                        <div className="relative inline-block max-w-full max-h-full">
                            <img
                                ref={imgRef}
                                src={imageSrc}
                                alt="Test Subject"
                                className="max-w-full max-h-[700px] rounded object-contain block"
                                onLoad={() => {
                                    // Trigger redraw if we have result
                                    if (result) {
                                        // Force re-render effect essentially
                                        setResult({ ...result });
                                    }
                                }}
                            />
                            <canvas
                                ref={canvasRef}
                                className="absolute inset-0 pointer-events-none"
                            />
                            {isProcessing && (
                                <div className="absolute inset-0 bg-black/60 flex items-center justify-center backdrop-blur-sm transition-all">
                                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-accent-primary"></div>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};
