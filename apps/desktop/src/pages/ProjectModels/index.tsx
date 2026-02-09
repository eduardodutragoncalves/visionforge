import { useCallback, useEffect, useMemo, useState } from 'react';
import { useParams } from 'react-router-dom';
import { Box, Cpu, HardDriveDownload, XCircle } from 'lucide-react';
import { Card } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { Select } from '../../components/ui/Select';
import { StatusPill } from '../../components/ui/StatusPill';
import { ENGINE_HTTP_URL } from '../../lib/engine';

type DeploymentTarget = {
    id: string;
    label: string;
    category: 'edge' | 'desktop';
    description: string;
};

const DEPLOYMENT_TARGETS: DeploymentTarget[] = [
    { id: 'edge_jetson', label: 'NVIDIA Jetson', category: 'edge', description: 'Jetson boards with CUDA/TensorRT ecosystem.' },
    { id: 'edge_raspberry_pi', label: 'Raspberry Pi', category: 'edge', description: 'ARM SBC baseline package using ONNX Runtime CPU.' },
    { id: 'edge_orange_pi', label: 'Orange Pi', category: 'edge', description: 'Generic Orange Pi package (ARM CPU baseline).' },
    { id: 'edge_generic', label: 'Generic Edge Device', category: 'edge', description: 'Fallback package for generic Linux edge targets.' },
    { id: 'desktop_linux', label: 'Desktop Linux', category: 'desktop', description: 'Linux workstation/server package.' },
    { id: 'desktop_windows', label: 'Desktop Windows', category: 'desktop', description: 'Windows package with PowerShell helper.' },
    { id: 'desktop_macos', label: 'Desktop macOS', category: 'desktop', description: 'macOS package with Python runner baseline.' }
];

interface DeploymentInfo {
    id?: string;
    target_id?: string;
    target_label?: string;
    format?: string;
    artifact_path?: string;
    created_at?: string;
    status?: string;
}

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
    onnx_exported?: boolean;
    exports?: {
        onnx?: boolean;
        onnx_path?: string | null;
    };
    deployments?: DeploymentInfo[];
}

type DeployState = {
    targetId: string;
    status: 'queued' | 'running';
    message?: string;
};

export const ProjectModels = () => {
    const { projectId } = useParams();
    const [variants, setVariants] = useState<Variant[]>([]);
    const [deployModalVariant, setDeployModalVariant] = useState<Variant | null>(null);
    const [selectedTargetId, setSelectedTargetId] = useState<string>('edge_jetson');
    const [deployingByVariant, setDeployingByVariant] = useState<Record<string, DeployState>>({});
    const [deployNotice, setDeployNotice] = useState<string | null>(null);

    const fetchVariants = useCallback(async () => {
        if (!projectId) return;
        const res = await fetch(`${ENGINE_HTTP_URL}/projects/${projectId}/variants`);
        if (res.ok) {
            setVariants(await res.json());
        }
    }, [projectId]);

    useEffect(() => {
        fetchVariants();
    }, [fetchVariants]);

    useEffect(() => {
        const onStarted = (e: Event) => {
            const detail = (e as CustomEvent).detail || {};
            const variantId = detail.variant_id;
            if (!variantId) return;
            setDeployingByVariant((prev) => ({
                ...prev,
                [variantId]: {
                    targetId: detail.target_id || 'edge_generic',
                    status: 'running',
                    message: 'Deployment started'
                }
            }));
        };
        const onProgress = (e: Event) => {
            const detail = (e as CustomEvent).detail || {};
            const variantId = detail.variant_id;
            if (!variantId) return;
            setDeployingByVariant((prev) => {
                const current = prev[variantId];
                if (!current) return prev;
                return {
                    ...prev,
                    [variantId]: {
                        ...current,
                        message: detail.message || current.message
                    }
                };
            });
        };
        const onCompleted = (e: Event) => {
            const detail = (e as CustomEvent).detail || {};
            const variantId = detail.variant_id;
            if (!variantId) return;
            setDeployingByVariant((prev) => {
                const next = { ...prev };
                delete next[variantId];
                return next;
            });
            setDeployNotice(`Deploy concluido para ${variantId} (${detail.target_id || 'target'}).`);
            fetchVariants();
        };
        const onFailed = (e: Event) => {
            const detail = (e as CustomEvent).detail || {};
            const variantId = detail.variant_id;
            if (variantId) {
                setDeployingByVariant((prev) => {
                    const next = { ...prev };
                    delete next[variantId];
                    return next;
                });
            }
            setDeployNotice(`Falha no deploy: ${detail.error || 'Unknown error'}`);
        };

        window.addEventListener('vf-export-started', onStarted as EventListener);
        window.addEventListener('vf-export-progress', onProgress as EventListener);
        window.addEventListener('vf-export-completed', onCompleted as EventListener);
        window.addEventListener('vf-export-failed', onFailed as EventListener);
        return () => {
            window.removeEventListener('vf-export-started', onStarted as EventListener);
            window.removeEventListener('vf-export-progress', onProgress as EventListener);
            window.removeEventListener('vf-export-completed', onCompleted as EventListener);
            window.removeEventListener('vf-export-failed', onFailed as EventListener);
        };
    }, [fetchVariants]);

    const selectedTarget = useMemo(
        () => DEPLOYMENT_TARGETS.find((target) => target.id === selectedTargetId),
        [selectedTargetId]
    );

    const handleDeploy = async () => {
        if (!projectId || !deployModalVariant) return;
        const variantId = deployModalVariant.id;
        setDeployNotice(null);
        setDeployingByVariant((prev) => ({
            ...prev,
            [variantId]: {
                targetId: selectedTargetId,
                status: 'queued',
                message: 'Queued'
            }
        }));
        setDeployModalVariant(null);

        const res = await fetch(`${ENGINE_HTTP_URL}/projects/${projectId}/variants/${variantId}/deploy`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ target_id: selectedTargetId })
        });
        if (!res.ok) {
            setDeployingByVariant((prev) => {
                const next = { ...prev };
                delete next[variantId];
                return next;
            });
            const text = await res.text();
            setDeployNotice(`Falha ao enfileirar deploy: ${text}`);
            return;
        }
    };

    const groupTargets = (category: 'edge' | 'desktop') =>
        DEPLOYMENT_TARGETS.filter((target) => target.category === category)
            .map((target) => ({ value: target.id, label: target.label }));

    return (
        <div className="project-models animate-fade-in space-y-6">
            <header className="flex justify-between items-center mb-6">
                <div>
                    <h2 className="text-2xl font-bold">Model Deployment</h2>
                    <p className="text-secondary">Choose the destination runtime and generate deployment artifacts for {projectId}</p>
                </div>
            </header>

            {deployNotice && (
                <Card className="border border-accent-primary/40 bg-accent-primary/10">
                    <div className="flex items-start justify-between gap-3">
                        <div className="text-sm">{deployNotice}</div>
                        <Button size="sm" variant="ghost" onClick={() => setDeployNotice(null)}>
                            <XCircle size={14} className="mr-1" />
                            Close
                        </Button>
                    </div>
                </Card>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {variants.length === 0 && (
                    <div className="col-span-full text-center text-secondary py-12">
                        No models trained yet. Go to "Train" to start a run.
                    </div>
                )}

                {variants.map((variant) => {
                    const deployments = variant.deployments || [];
                    const activeDeploy = deployingByVariant[variant.id];
                    const deployedTargets = Array.from(
                        new Set(deployments.map((deployment) => deployment.target_id).filter(Boolean))
                    ) as string[];
                    return (
                        <Card key={variant.id} className="hover:border-accent-primary/50 transition-colors">
                            <div className="flex justify-between items-start mb-4">
                                <div>
                                    <h3 className="font-bold text-lg">{variant.name}</h3>
                                    <div className="text-xs text-secondary font-mono">{variant.id}</div>
                                </div>
                                <StatusPill status="success" label="READY" />
                            </div>

                            <div className="space-y-2 mb-5">
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
                                <div className="text-xs text-secondary uppercase tracking-[0.24em] mt-2">Deploy Targets</div>
                                {deployedTargets.length === 0 ? (
                                    <div className="text-xs text-secondary">No deployments generated yet.</div>
                                ) : (
                                    <div className="flex flex-wrap gap-2">
                                        {deployedTargets.map((targetId) => (
                                            <span key={`${variant.id}-${targetId}`} className="stat-chip">
                                                {targetId}
                                            </span>
                                        ))}
                                    </div>
                                )}
                            </div>

                            {activeDeploy && (
                                <div className="mb-4 rounded border border-accent-warning/50 bg-accent-warning/10 px-3 py-2 text-xs text-accent-warning">
                                    Deploying to <strong>{activeDeploy.targetId}</strong> · {activeDeploy.message || activeDeploy.status}
                                </div>
                            )}

                            <div className="grid grid-cols-2 gap-2">
                                <Button size="sm" variant="secondary">Test</Button>
                                <Button
                                    size="sm"
                                    variant="secondary"
                                    onClick={() => {
                                        setSelectedTargetId('edge_jetson');
                                        setDeployModalVariant(variant);
                                    }}
                                    disabled={Boolean(activeDeploy)}
                                >
                                    <HardDriveDownload size={14} className="mr-1" />
                                    Deploy
                                </Button>
                            </div>
                        </Card>
                    );
                })}
            </div>

            {deployModalVariant && (
                <div className="modal-backdrop">
                    <div className="modal-card max-w-2xl">
                        <div className="modal-header mb-4">
                            <div>
                                <h3 className="text-xl font-semibold">Deployment Target</h3>
                                <p className="text-secondary text-sm mt-1">Variant: {deployModalVariant.name}</p>
                            </div>
                            <Button size="sm" variant="ghost" onClick={() => setDeployModalVariant(null)}>
                                <XCircle size={14} className="mr-1" />
                                Close
                            </Button>
                        </div>
                        <Card>
                            <div className="flex items-center gap-2 mb-3">
                                <Cpu size={16} className="text-accent-primary" />
                                <div className="text-sm font-semibold uppercase tracking-[0.24em]">Target Selection</div>
                            </div>
                            <Select
                                label="Destination"
                                options={DEPLOYMENT_TARGETS.map((target) => ({
                                    value: target.id,
                                    label: `${target.category === 'edge' ? 'Edge' : 'Desktop'} · ${target.label}`
                                }))}
                                value={selectedTargetId}
                                onChange={(event) => setSelectedTargetId(event.target.value)}
                            />
                            <div className="mt-3 flex flex-wrap gap-2 text-[11px] uppercase tracking-[0.2em] text-secondary">
                                <span className="stat-chip">Edge: <strong>{groupTargets('edge').length}</strong></span>
                                <span className="stat-chip">Desktop: <strong>{groupTargets('desktop').length}</strong></span>
                            </div>
                        </Card>

                        <Card className="mt-4 border border-graphite-700/70 bg-graphite-900/60">
                            <div className="flex items-center gap-2 mb-2">
                                <Box size={16} className="text-accent-primary" />
                                <div className="text-sm font-semibold uppercase tracking-[0.24em]">Selected target profile</div>
                            </div>
                            <div className="text-sm">{selectedTarget?.label || selectedTargetId}</div>
                            <div className="text-xs text-secondary mt-1">{selectedTarget?.description}</div>
                            <div className="text-xs text-secondary mt-2">
                                Initial profile generates ONNX artifact + target-specific deployment docs/scripts.
                            </div>
                        </Card>

                        <div className="mt-5 flex justify-end gap-2">
                            <Button variant="ghost" onClick={() => setDeployModalVariant(null)}>Cancel</Button>
                            <Button variant="primary" onClick={handleDeploy}>
                                <HardDriveDownload size={14} className="mr-1" />
                                Generate Deployment Package
                            </Button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};
