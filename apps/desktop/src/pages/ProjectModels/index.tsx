import { useCallback, useEffect, useMemo, useState } from 'react';
import { useParams } from 'react-router-dom';
import { Box, CheckCircle2, Cpu, ExternalLink, HardDriveDownload, Loader2, XCircle } from 'lucide-react';
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
    package_dir?: string;
    artifact_path?: string;
    readme_path?: string;
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
    stage?: string;
    progress?: number;
    startedAt: number;
};

type DeploymentInspect = {
    variant: Variant;
    deployment: DeploymentInfo;
};

const targetLabel = (targetId?: string) => {
    if (!targetId) return 'Unknown target';
    return DEPLOYMENT_TARGETS.find((target) => target.id === targetId)?.label || targetId;
};

const deriveProgress = (detail: any, fallback: DeployState | undefined): number => {
    if (typeof detail?.progress === 'number') {
        return Math.max(0, Math.min(100, detail.progress));
    }
    const msg = String(detail?.message || '').toLowerCase();
    if (msg.includes('loading')) return 15;
    if (msg.includes('exporting')) return 55;
    if (msg.includes('package')) return 75;
    if (msg.includes('script')) return 88;
    if (msg.includes('manifest')) return 96;
    return fallback?.progress ?? (fallback?.status === 'queued' ? 5 : 35);
};

export const ProjectModels = () => {
    const { projectId } = useParams();
    const [variants, setVariants] = useState<Variant[]>([]);
    const [deployModalVariant, setDeployModalVariant] = useState<Variant | null>(null);
    const [deployInspect, setDeployInspect] = useState<DeploymentInspect | null>(null);
    const [selectedTargetId, setSelectedTargetId] = useState<string>('edge_jetson');
    const [deployingByVariant, setDeployingByVariant] = useState<Record<string, DeployState>>({});
    const [deployNotice, setDeployNotice] = useState<string | null>(null);
    const [nowTs, setNowTs] = useState<number>(Date.now());

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
        const timer = window.setInterval(() => setNowTs(Date.now()), 1000);
        return () => window.clearInterval(timer);
    }, []);

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
                    message: 'Deployment started',
                    stage: detail.stage || 'start',
                    progress: 12,
                    startedAt: prev[variantId]?.startedAt || Date.now()
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
                        message: detail.message || current.message,
                        stage: detail.stage || current.stage,
                        progress: deriveProgress(detail, current),
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
            setDeployNotice(`Deploy concluido para ${variantId} (${targetLabel(detail.target_id)}).`);
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
                message: 'Queued',
                stage: 'queue',
                progress: 5,
                startedAt: Date.now()
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

    const activeDeployEntries = useMemo(
        () => Object.entries(deployingByVariant),
        [deployingByVariant]
    );

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

            {activeDeployEntries.length > 0 && (
                <Card className="border border-accent-warning/50 bg-accent-warning/10">
                    <div className="text-xs uppercase tracking-[0.28em] text-accent-warning mb-2">Deploy progress</div>
                    <div className="space-y-3">
                        {activeDeployEntries.map(([variantId, state]) => {
                            const elapsedSec = Math.max(0, Math.floor((nowTs - state.startedAt) / 1000));
                            const progress = Math.max(3, Math.min(97, state.progress ?? 0));
                            return (
                                <div key={`deploying-${variantId}`} className="rounded border border-accent-warning/40 bg-graphite-900/40 px-3 py-2">
                                    <div className="flex items-center justify-between gap-2 text-xs">
                                        <div className="flex items-center gap-2">
                                            <Loader2 size={12} className="animate-spin" />
                                            <span>{variantId}</span>
                                            <span className="text-secondary">→ {targetLabel(state.targetId)}</span>
                                        </div>
                                        <div className="text-secondary">{elapsedSec}s</div>
                                    </div>
                                    <div className="mt-1 text-[11px] text-secondary">{state.message || state.status}</div>
                                    <div className="mt-2 h-1.5 rounded bg-graphite-700 overflow-hidden">
                                        <div className="h-full bg-accent-warning transition-all duration-500" style={{ width: `${progress}%` }} />
                                    </div>
                                </div>
                            );
                        })}
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
                                {deployments.length === 0 ? (
                                    <div className="text-xs text-secondary">No deployments generated yet.</div>
                                ) : (
                                    <div className="flex flex-wrap gap-2">
                                        {deployedTargets.map((targetId) => {
                                            const latest = [...deployments]
                                                .filter((deployment) => deployment.target_id === targetId)
                                                .sort((a, b) => String(b.created_at || '').localeCompare(String(a.created_at || '')))[0];
                                            return (
                                                <button
                                                    key={`${variant.id}-${targetId}`}
                                                    type="button"
                                                    className="tab-button tab-button-active"
                                                    onClick={() => {
                                                        if (latest) setDeployInspect({ variant, deployment: latest });
                                                    }}
                                                    title="Open deployment details"
                                                >
                                                    {targetLabel(targetId)}
                                                </button>
                                            );
                                        })}
                                    </div>
                                )}
                            </div>

                            {activeDeploy && (
                                <div className="mb-4 rounded border border-accent-warning/50 bg-accent-warning/10 px-3 py-2 text-xs text-accent-warning">
                                    <div className="flex items-center justify-between gap-2">
                                        <span className="inline-flex items-center gap-2">
                                            <Loader2 size={12} className="animate-spin" />
                                            Deploying to <strong>{targetLabel(activeDeploy.targetId)}</strong>
                                        </span>
                                        <span>{Math.max(3, Math.min(97, activeDeploy.progress ?? 0))}%</span>
                                    </div>
                                    <div className="mt-1 text-[11px]">{activeDeploy.message || activeDeploy.status}</div>
                                    <div className="mt-2 h-1.5 rounded bg-graphite-800 overflow-hidden">
                                        <div
                                            className="h-full bg-accent-warning transition-all duration-500"
                                            style={{ width: `${Math.max(3, Math.min(97, activeDeploy.progress ?? 0))}%` }}
                                        />
                                    </div>
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

            {deployInspect && (
                <div className="modal-backdrop">
                    <div className="modal-card max-w-2xl">
                        <div className="modal-header mb-4">
                            <div>
                                <h3 className="text-xl font-semibold">Deployment Artifacts</h3>
                                <p className="text-secondary text-sm mt-1">
                                    {deployInspect.variant.name} · {targetLabel(deployInspect.deployment.target_id)}
                                </p>
                            </div>
                            <Button size="sm" variant="ghost" onClick={() => setDeployInspect(null)}>
                                <XCircle size={14} className="mr-1" />
                                Close
                            </Button>
                        </div>
                        <div className="space-y-3 text-sm">
                            <div className="rounded border border-graphite-700/70 bg-graphite-900 px-3 py-2">
                                <div className="text-xs uppercase tracking-[0.24em] text-secondary">Status</div>
                                <div className="mt-1 inline-flex items-center gap-2">
                                    <CheckCircle2 size={14} className="text-success" />
                                    <span>{deployInspect.deployment.status || 'completed'}</span>
                                </div>
                            </div>
                            <div className="rounded border border-graphite-700/70 bg-graphite-900 px-3 py-2">
                                <div className="text-xs uppercase tracking-[0.24em] text-secondary">Format</div>
                                <div className="mt-1">{deployInspect.deployment.format || 'onnx'}</div>
                            </div>
                            <div className="rounded border border-graphite-700/70 bg-graphite-900 px-3 py-2">
                                <div className="text-xs uppercase tracking-[0.24em] text-secondary">Artifact Path</div>
                                <div className="mt-1 font-mono text-xs break-all">{deployInspect.deployment.artifact_path || 'Not available'}</div>
                            </div>
                            <div className="rounded border border-graphite-700/70 bg-graphite-900 px-3 py-2">
                                <div className="text-xs uppercase tracking-[0.24em] text-secondary">Package Folder</div>
                                <div className="mt-1 font-mono text-xs break-all">{deployInspect.deployment.package_dir || 'Not available'}</div>
                            </div>
                        </div>
                        <div className="mt-5 flex flex-wrap justify-end gap-2">
                            <Button
                                variant="ghost"
                                onClick={async () => {
                                    const artifact = deployInspect.deployment.artifact_path;
                                    if (!artifact) return;
                                    const result = await window.ipcRenderer?.revealPath?.(artifact);
                                    if (result?.ok === false) {
                                        setDeployNotice(`Nao foi possivel abrir artefato: ${result.message || 'unknown error'}`);
                                    }
                                }}
                            >
                                <ExternalLink size={14} className="mr-1" />
                                Reveal Artifact
                            </Button>
                            <Button
                                variant="secondary"
                                onClick={async () => {
                                    const pkg = deployInspect.deployment.package_dir;
                                    if (!pkg) return;
                                    const result = await window.ipcRenderer?.openPath?.(pkg);
                                    if (result?.ok === false) {
                                        setDeployNotice(`Nao foi possivel abrir pasta: ${result.message || 'unknown error'}`);
                                    }
                                }}
                            >
                                <ExternalLink size={14} className="mr-1" />
                                Open Package Folder
                            </Button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};
