import { useState, useRef, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { Card } from '../../components/ui/Card';
import { Select } from '../../components/ui/Select';
import { Upload, Camera, Zap } from 'lucide-react';
import { ENGINE_HTTP_URL } from '../../lib/engine';

interface Prediction {
    box: number[]; // x1, y1, x2, y2
    confidence: number;
    class_name: string;
    class_id: number;
}

interface InferenceResult {
    predictions: Prediction[];
    latency_ms: number;
    fps_est: number;
}

export const ProjectTest = () => {
    const { projectId } = useParams();

    // State
    const [selectedVariant, setSelectedVariant] = useState<string>("");
    const [variants, setVariants] = useState<{ id: string, name: string }[]>([]);
    const [imageSrc, setImageSrc] = useState<string | null>(null);
    const [result, setResult] = useState<InferenceResult | null>(null);
    const [isProcessing, setIsProcessing] = useState(false);

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

    const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            const url = URL.createObjectURL(file);
            setImageSrc(url);
            setResult(null); // Clear previous results

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
            } else {
                console.error("Inference failed");
            }
        } catch (e) {
            console.error(e);
        } finally {
            setIsProcessing(false);
        }
    };

    // Draw Overlay
    useEffect(() => {
        if (imageSrc && result && imgRef.current && canvasRef.current) {
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

    return (
        <div className="project-test animate-fade-in space-y-6 h-full flex flex-col">
            <header className="flex justify-between items-center">
                <div>
                    <h2 className="text-2xl font-bold">Model Playground</h2>
                    <p className="text-secondary">Test your trained models with local images</p>
                </div>
            </header>

            <div className="grid grid-cols-12 gap-6 flex-1">
                {/* Control Panel */}
                <Card className="col-span-3 h-fit space-y-6">
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
                        </div>
                    )}
                </Card>

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
