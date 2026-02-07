import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { Card } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { StatusPill } from '../../components/ui/StatusPill';
import { ENGINE_HTTP_URL } from '../../lib/engine';

interface Variant {
    id: string;
    name: string;
    created_at: string;
    base_model: string;
    metrics: {
        final_map?: number;
        accuracy?: number;
        loss?: number;
    };
    run_id: string;
}

export const ProjectModels = () => {
    const { projectId } = useParams();
    const [variants, setVariants] = useState<Variant[]>([]);
    const [exportingId, setExportingId] = useState<string | null>(null);
    const [exportResult, setExportResult] = useState<string | null>(null);

    useEffect(() => {
        const onExportComplete = (e: CustomEvent) => {
            setExportResult(e.detail.path);
        };
        window.addEventListener('vf-export-completed' as any, onExportComplete as any);
        return () => window.removeEventListener('vf-export-completed' as any, onExportComplete as any);
    }, []);

    const handleExport = async (variantId: string) => {
        setExportingId(variantId);
        setExportResult(null);
        await fetch(`${ENGINE_HTTP_URL}/projects/${projectId}/variants/${variantId}/export`, {
            method: 'POST'
        });
    };


    useEffect(() => {
        const fetchVariants = async () => {
            const res = await fetch(`${ENGINE_HTTP_URL}/projects/${projectId}/variants`);
            if (res.ok) {
                setVariants(await res.json());
            }
        };
        fetchVariants();
    }, [projectId]);

    return (
        <div className="project-models animate-fade-in space-y-6">
            <header className="flex justify-between items-center mb-6">
                <div>
                    <h2 className="text-2xl font-bold">Trained Models</h2>
                    <p className="text-secondary">Manage versions and variants for {projectId}</p>
                </div>
            </header>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {variants.length === 0 && (
                    <div className="col-span-full text-center text-secondary py-12">
                        No models trained yet. Go to "Train" to start a run.
                    </div>
                )}

                {variants.map(variant => (
                    <Card key={variant.id} className="hover:border-accent-primary/50 transition-colors">
                        <div className="flex justify-between items-start mb-4">
                            <div>
                                <h3 className="font-bold text-lg">{variant.name}</h3>
                                <div className="text-xs text-secondary font-mono">{variant.id}</div>
                            </div>
                            <StatusPill status="success" label="READY" />
                        </div>

                        <div className="space-y-2 mb-6">
                            <div className="flex justify-between text-sm">
                                <span className="text-secondary">Base</span>
                                <span>{variant.base_model}</span>
                            </div>
                            <div className="flex justify-between text-sm">
                                <span className="text-secondary">mAP50</span>
                                <span className="text-accent-primary font-bold">
                                    {(variant.metrics.final_map || 0).toFixed(4)}
                                </span>
                            </div>
                            <div className="flex justify-between text-sm">
                                <span className="text-secondary">Loss</span>
                                <span>{(variant.metrics.loss || 0).toFixed(4)}</span>
                            </div>
                        </div>

                        <div className="grid grid-cols-2 gap-2">
                            <Button size="sm" variant="secondary">Test</Button>
                            <Button
                                size="sm"
                                variant="secondary"
                                onClick={() => handleExport(variant.id)}
                            >
                                Export ONNX
                            </Button>
                        </div>
                    </Card>
                ))}
            </div>

            {/* Export Progress / Result Modal or Toast area */}
            {exportingId && (
                <div className="fixed bottom-4 right-4 bg-gray-900 border border-accent-primary p-4 rounded shadow-lg animate-fade-in max-w-sm">
                    <div className="font-bold mb-1">Exporting Model...</div>
                    <div className="text-xs text-secondary mb-2">{exportingId}</div>
                    {exportResult ? (
                        <div className="text-success text-sm break-all">
                            <div>Done!</div>
                            <div className="font-mono text-xs mt-1">{exportResult}</div>
                            <Button size="sm" variant="ghost" className="mt-2 h-6" onClick={() => setExportResult(null)}>Close</Button>
                        </div>
                    ) : (
                        <div className="flex items-center gap-2 text-sm">
                            <div className="animate-spin h-3 w-3 border-b-2 border-white rounded-full"></div>
                            Processing...
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};
