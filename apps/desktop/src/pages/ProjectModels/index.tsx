import { useCallback, useEffect, useMemo, useState } from 'react';
import { useParams } from 'react-router-dom';
import type { LucideIcon } from 'lucide-react';
import {
    Activity,
    Box,
    CheckCircle2,
    Clock3,
    Cpu,
    ExternalLink,
    HardDriveDownload,
    LayoutGrid,
    Loader2,
    Monitor,
    Rocket,
    Server,
    Target,
    XCircle
} from 'lucide-react';
import { Card } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
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

interface SummaryMetricCardProps {
    title: string;
    value: string;
    hint: string;
    icon: LucideIcon;
    tone?: 'default' | 'success' | 'warning' | 'running';
}

const targetLabel = (targetId?: string) => {
    if (!targetId) return 'Unknown target';
    return DEPLOYMENT_TARGETS.find((target) => target.id === targetId)?.label || targetId;
};

const targetIcon = (targetId: string): LucideIcon => {
    if (targetId.startsWith('edge_')) return Cpu;
    return Monitor;
};

const formatMetric = (value?: number, decimals = 4) => {
    if (typeof value !== 'number' || Number.isNaN(value)) return '0.0000';
    return value.toFixed(decimals);
};

const formatDate = (value?: string) => {
    if (!value) return 'Unknown date';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleString();
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

const SummaryMetricCard = ({ title, value, hint, icon: Icon, tone = 'default' }: SummaryMetricCardProps) => {
    const toneClass =
        tone === 'success'
            ? 'border-accent-success/35 bg-accent-success/5'
            : tone === 'warning'
                ? 'border-accent-warning/35 bg-accent-warning/5'
                : tone === 'running'
                    ? 'border-accent-primary/35 bg-accent-primary/5'
                    : 'border-graphite-700/70 bg-graphite-900/55';

    return (
        <Card className={`border ${toneClass} p-4`}>
            <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.3em] text-secondary">
                <span>{title}</span>
                <Icon size={14} />
            </div>
            <div className="mt-2 text-2xl font-mono">{value}</div>
            <div className="mt-1 text-xs text-secondary">{hint}</div>
        </Card>
    );
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
                        progress: deriveProgress(detail, current)
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

    const variantsSorted = useMemo(
        () => [...variants].sort((a, b) => String(b.created_at || '').localeCompare(String(a.created_at || ''))),
        [variants]
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
        }
    };

    const activeDeployEntries = useMemo(
        () => Object.entries(deployingByVariant),
        [deployingByVariant]
    );

    const totalDeployments = useMemo(
        () => variants.reduce((count, variant) => count + (variant.deployments?.length || 0), 0),
        [variants]
    );

    const deployedModelCount = useMemo(
        () => variants.filter((variant) => (variant.deployments?.length || 0) > 0).length,
        [variants]
    );

    const edgeTargetCount = useMemo(
        () =>
            variants.reduce((count, variant) => {
                const uniqueTargets = new Set((variant.deployments || []).map((deployment) => deployment.target_id));
                return count + Array.from(uniqueTargets).filter((targetId) => String(targetId || '').startsWith('edge_')).length;
            }, 0),
        [variants]
    );

    const desktopTargetCount = useMemo(
        () =>
            variants.reduce((count, variant) => {
                const uniqueTargets = new Set((variant.deployments || []).map((deployment) => deployment.target_id));
                return count + Array.from(uniqueTargets).filter((targetId) => String(targetId || '').startsWith('desktop_')).length;
            }, 0),
        [variants]
    );

    const edgeTargets = useMemo(
        () => DEPLOYMENT_TARGETS.filter((target) => target.category === 'edge'),
        []
    );

    const desktopTargets = useMemo(
        () => DEPLOYMENT_TARGETS.filter((target) => target.category === 'desktop'),
        []
    );

    return (
        <div className="project-models animate-fade-in w-full max-w-none space-y-6 pb-12">
            <header className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
                <div className="flex items-start gap-3">
                    <div className="rounded-xl border border-accent-primary/30 bg-accent-primary/10 p-2">
                        <LayoutGrid size={20} className="text-accent-primary" />
                    </div>
                    <div>
                        <h2 className="text-3xl font-bold">Model Deployment</h2>
                        <p className="text-secondary">
                            Organize model outputs and generate deployment packages for project {projectId}.
                        </p>
                    </div>
                </div>
                <StatusPill status="running" label={`MODELS: ${variants.length}`} pulsing={activeDeployEntries.length > 0} />
            </header>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
                <SummaryMetricCard
                    title="Trained Models"
                    value={String(variants.length)}
                    hint="Variants available in this project"
                    icon={Box}
                />
                <SummaryMetricCard
                    title="Deployments"
                    value={String(totalDeployments)}
                    hint={`${deployedModelCount} models with at least one target`}
                    icon={Server}
                    tone="success"
                />
                <SummaryMetricCard
                    title="In Progress"
                    value={String(activeDeployEntries.length)}
                    hint="Deployment jobs running right now"
                    icon={Rocket}
                    tone="running"
                />
                <SummaryMetricCard
                    title="Targets Used"
                    value={String(edgeTargetCount + desktopTargetCount)}
                    hint={`Edge: ${edgeTargetCount} · Desktop: ${desktopTargetCount}`}
                    icon={Target}
                    tone="warning"
                />
            </div>

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
                    <div className="mb-3 flex items-center justify-between gap-3">
                        <div className="flex items-center gap-2 text-xs uppercase tracking-[0.28em] text-accent-warning">
                            <Clock3 size={14} />
                            Deploy progress
                        </div>
                        <StatusPill status="warning" label={`${activeDeployEntries.length} ACTIVE`} />
                    </div>
                    <div className="space-y-2.5">
                        {activeDeployEntries.map(([variantId, state]) => {
                            const elapsedSec = Math.max(0, Math.floor((nowTs - state.startedAt) / 1000));
                            const progress = Math.max(5, Math.min(97, state.progress ?? 0));
                            return (
                                <div key={`deploying-${variantId}`} className="rounded-xl border border-accent-warning/40 bg-graphite-900/45 px-4 py-3">
                                    <div className="flex items-center justify-between gap-2 text-xs uppercase tracking-[0.2em]">
                                        <div className="flex items-center gap-2">
                                            <Loader2 size={12} className="animate-spin" />
                                            <span className="font-semibold text-bone">{variantId}</span>
                                            <span className="text-secondary">to {targetLabel(state.targetId)}</span>
                                        </div>
                                        <div className="text-secondary">{elapsedSec}s · {progress}%</div>
                                    </div>
                                    <div className="mt-1 text-xs text-secondary">{state.message || state.status}</div>
                                    <div className="mt-2 h-2 overflow-hidden rounded bg-graphite-700">
                                        <div className="h-full bg-accent-warning transition-all duration-500" style={{ width: `${progress}%` }} />
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </Card>
            )}

            <div className="grid grid-cols-1 gap-4 xl:grid-cols-2 2xl:grid-cols-3">
                {variantsSorted.length === 0 && (
                    <Card className="col-span-full border border-graphite-700/70 bg-graphite-900/40 py-14">
                        <div className="mx-auto max-w-md text-center text-secondary">
                            <Activity size={20} className="mx-auto mb-3 text-haze" />
                            <div className="mb-2 text-sm uppercase tracking-[0.3em]">No models found</div>
                            <div className="text-sm">
                                Train a model first, then come back here to run tests and generate deployment bundles.
                            </div>
                        </div>
                    </Card>
                )}

                {variantsSorted.map((variant) => {
                    const deployments = variant.deployments || [];
                    const activeDeploy = deployingByVariant[variant.id];
                    const deployedTargets = Array.from(
                        new Set(
                            deployments
                                .map((deployment) => deployment.target_id)
                                .filter((targetId): targetId is string => Boolean(targetId))
                        )
                    );

                    const status = activeDeploy ? 'running' : deployments.length > 0 ? 'success' : 'idle';
                    const statusLabel = activeDeploy ? 'DEPLOYING' : deployments.length > 0 ? 'DEPLOYED' : 'READY';
                    const statusHint = activeDeploy
                        ? `${Math.max(5, Math.min(97, activeDeploy.progress ?? 0))}% · ${activeDeploy.message || activeDeploy.status}`
                        : deployments.length > 0
                            ? `${deployments.length} package${deployments.length > 1 ? 's' : ''} generated`
                            : 'No deployments generated yet';

                    return (
                        <Card
                            key={variant.id}
                            className="flex h-full flex-col border border-graphite-700/70 bg-gradient-to-b from-graphite-900/80 to-graphite-900/50 p-5 transition-all duration-150 hover:border-accent-primary/40"
                        >
                            <div className="flex items-start justify-between gap-3">
                                <div className="min-w-0">
                                    <h3 className="truncate text-xl font-semibold">{variant.name}</h3>
                                    <div className="mt-1 truncate font-mono text-xs text-secondary">{variant.id}</div>
                                </div>
                                <StatusPill status={status} label={statusLabel} pulsing={Boolean(activeDeploy)} />
                            </div>

                            <div className="mt-4 grid grid-cols-3 gap-2 text-xs">
                                <div className="rounded-lg border border-graphite-700/70 bg-graphite-900/70 px-2 py-2">
                                    <div className="uppercase tracking-[0.24em] text-secondary">mAP50</div>
                                    <div className="mt-1 font-mono text-accent-primary">{formatMetric(variant.metrics.final_map)}</div>
                                </div>
                                <div className="rounded-lg border border-graphite-700/70 bg-graphite-900/70 px-2 py-2">
                                    <div className="uppercase tracking-[0.24em] text-secondary">Accuracy</div>
                                    <div className="mt-1 font-mono">{formatMetric(variant.metrics.accuracy)}</div>
                                </div>
                                <div className="rounded-lg border border-graphite-700/70 bg-graphite-900/70 px-2 py-2">
                                    <div className="uppercase tracking-[0.24em] text-secondary">Loss</div>
                                    <div className="mt-1 font-mono">{formatMetric(variant.metrics.loss)}</div>
                                </div>
                            </div>

                            <div className="mt-4 space-y-2 text-sm">
                                <div className="flex items-center justify-between">
                                    <span className="text-secondary">Base model</span>
                                    <span className="font-medium">{variant.base_model}</span>
                                </div>
                                <div className="flex items-center justify-between">
                                    <span className="text-secondary">Created</span>
                                    <span className="text-xs">{formatDate(variant.created_at)}</span>
                                </div>
                            </div>

                            <div className="mt-4 rounded-lg border border-graphite-700/70 bg-graphite-900/55 px-3 py-2">
                                <div className="text-[11px] uppercase tracking-[0.3em] text-secondary">Deployment State</div>
                                <div className="mt-1 text-sm">{statusHint}</div>
                                {activeDeploy && (
                                    <div className="mt-2 h-1.5 overflow-hidden rounded bg-graphite-700">
                                        <div
                                            className="h-full bg-accent-warning transition-all duration-500"
                                            style={{ width: `${Math.max(5, Math.min(97, activeDeploy.progress ?? 0))}%` }}
                                        />
                                    </div>
                                )}
                            </div>

                            <div className="mt-4">
                                <div className="mb-2 text-[11px] uppercase tracking-[0.32em] text-secondary">Deploy Targets</div>
                                {deployedTargets.length === 0 ? (
                                    <div className="rounded-lg border border-dashed border-graphite-700/80 px-3 py-2 text-xs text-secondary">
                                        No target exported yet. Click <strong>Deploy</strong> to generate the first package.
                                    </div>
                                ) : (
                                    <div className="flex flex-wrap gap-2">
                                        {deployedTargets.map((targetId) => {
                                            const latest = [...deployments]
                                                .filter((deployment) => deployment.target_id === targetId)
                                                .sort((a, b) => String(b.created_at || '').localeCompare(String(a.created_at || '')))[0];
                                            const TargetIcon = targetIcon(targetId);

                                            return (
                                                <button
                                                    key={`${variant.id}-${targetId}`}
                                                    type="button"
                                                    className="tab-button tab-button-active inline-flex items-center gap-1.5"
                                                    onClick={() => {
                                                        if (latest) setDeployInspect({ variant, deployment: latest });
                                                    }}
                                                    title="Open deployment details"
                                                >
                                                    <TargetIcon size={11} />
                                                    {targetLabel(targetId)}
                                                </button>
                                            );
                                        })}
                                    </div>
                                )}
                            </div>

                            <div className="mt-5 grid grid-cols-2 gap-2">
                                <Button size="sm" variant="secondary">Test</Button>
                                <Button
                                    size="sm"
                                    variant={activeDeploy ? 'ghost' : 'primary'}
                                    onClick={() => {
                                        setSelectedTargetId('edge_jetson');
                                        setDeployModalVariant(variant);
                                    }}
                                    disabled={Boolean(activeDeploy)}
                                >
                                    <HardDriveDownload size={14} className="mr-1" />
                                    {activeDeploy ? 'Deploying...' : 'Deploy'}
                                </Button>
                            </div>
                        </Card>
                    );
                })}
            </div>

            {deployModalVariant && (
                <div className="modal-backdrop">
                    <div className="modal-card max-w-4xl">
                        <div className="modal-header mb-4">
                            <div>
                                <h3 className="text-2xl font-semibold">Deployment Target</h3>
                                <p className="mt-1 text-sm text-secondary">Variant: {deployModalVariant.name}</p>
                            </div>
                            <Button size="sm" variant="ghost" onClick={() => setDeployModalVariant(null)}>
                                <XCircle size={14} className="mr-1" />
                                Close
                            </Button>
                        </div>

                        <div className="grid grid-cols-1 gap-4 xl:grid-cols-12">
                            <Card className="xl:col-span-8 border border-graphite-700/70 bg-graphite-900/55">
                                <div className="mb-3 flex items-center gap-2">
                                    <Cpu size={16} className="text-accent-primary" />
                                    <div className="text-sm font-semibold uppercase tracking-[0.24em]">Choose destination</div>
                                </div>
                                <div className="space-y-4">
                                    <div>
                                        <div className="mb-2 text-xs uppercase tracking-[0.28em] text-secondary">Edge devices</div>
                                        <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
                                            {edgeTargets.map((target) => {
                                                const selected = selectedTargetId === target.id;
                                                return (
                                                    <button
                                                        key={target.id}
                                                        type="button"
                                                        className={`rounded-lg border px-3 py-3 text-left transition-all ${
                                                            selected
                                                                ? 'border-accent-primary bg-accent-primary/10 text-bone'
                                                                : 'border-graphite-700/70 bg-graphite-900/70 text-secondary hover:border-accent-primary/40'
                                                        }`}
                                                        onClick={() => setSelectedTargetId(target.id)}
                                                    >
                                                        <div className="flex items-center gap-2 text-sm font-semibold">
                                                            <Cpu size={14} className={selected ? 'text-accent-primary' : 'text-secondary'} />
                                                            {target.label}
                                                        </div>
                                                        <div className="mt-1 text-xs">{target.description}</div>
                                                    </button>
                                                );
                                            })}
                                        </div>
                                    </div>
                                    <div>
                                        <div className="mb-2 text-xs uppercase tracking-[0.28em] text-secondary">Desktop runtimes</div>
                                        <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
                                            {desktopTargets.map((target) => {
                                                const selected = selectedTargetId === target.id;
                                                return (
                                                    <button
                                                        key={target.id}
                                                        type="button"
                                                        className={`rounded-lg border px-3 py-3 text-left transition-all ${
                                                            selected
                                                                ? 'border-accent-primary bg-accent-primary/10 text-bone'
                                                                : 'border-graphite-700/70 bg-graphite-900/70 text-secondary hover:border-accent-primary/40'
                                                        }`}
                                                        onClick={() => setSelectedTargetId(target.id)}
                                                    >
                                                        <div className="flex items-center gap-2 text-sm font-semibold">
                                                            <Monitor size={14} className={selected ? 'text-accent-primary' : 'text-secondary'} />
                                                            {target.label}
                                                        </div>
                                                        <div className="mt-1 text-xs">{target.description}</div>
                                                    </button>
                                                );
                                            })}
                                        </div>
                                    </div>
                                </div>
                            </Card>

                            <Card className="xl:col-span-4 border border-accent-primary/30 bg-accent-primary/5">
                                <div className="mb-2 flex items-center gap-2">
                                    <Server size={16} className="text-accent-primary" />
                                    <div className="text-sm font-semibold uppercase tracking-[0.24em]">Profile preview</div>
                                </div>
                                <div className="text-sm font-semibold">{selectedTarget?.label || selectedTargetId}</div>
                                <div className="mt-1 text-xs text-secondary">{selectedTarget?.description}</div>
                                <div className="mt-4 space-y-2 text-xs text-secondary">
                                    <div className="rounded border border-graphite-700/70 bg-graphite-900/70 px-2 py-2">
                                        ONNX artifact is generated automatically.
                                    </div>
                                    <div className="rounded border border-graphite-700/70 bg-graphite-900/70 px-2 py-2">
                                        Includes setup scripts and target README.
                                    </div>
                                    <div className="rounded border border-graphite-700/70 bg-graphite-900/70 px-2 py-2">
                                        Deployment appears in target chips after completion.
                                    </div>
                                </div>
                            </Card>
                        </div>

                        <div className="mt-5 flex flex-wrap justify-end gap-2">
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
                                <h3 className="text-2xl font-semibold">Deployment Artifacts</h3>
                                <p className="mt-1 text-sm text-secondary">
                                    {deployInspect.variant.name} · {targetLabel(deployInspect.deployment.target_id)}
                                </p>
                            </div>
                            <Button size="sm" variant="ghost" onClick={() => setDeployInspect(null)}>
                                <XCircle size={14} className="mr-1" />
                                Close
                            </Button>
                        </div>

                        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                            <div className="rounded-xl border border-accent-success/30 bg-accent-success/5 px-3 py-3">
                                <div className="text-xs uppercase tracking-[0.24em] text-secondary">Status</div>
                                <div className="mt-1 inline-flex items-center gap-2 text-sm">
                                    <CheckCircle2 size={14} className="text-success" />
                                    <span>{deployInspect.deployment.status || 'completed'}</span>
                                </div>
                            </div>
                            <div className="rounded-xl border border-graphite-700/70 bg-graphite-900 px-3 py-3">
                                <div className="text-xs uppercase tracking-[0.24em] text-secondary">Format</div>
                                <div className="mt-1 text-sm">{deployInspect.deployment.format || 'onnx'}</div>
                            </div>
                            <div className="rounded-xl border border-graphite-700/70 bg-graphite-900 px-3 py-3 sm:col-span-2">
                                <div className="text-xs uppercase tracking-[0.24em] text-secondary">Artifact Path</div>
                                <div className="mt-1 break-all font-mono text-xs">{deployInspect.deployment.artifact_path || 'Not available'}</div>
                            </div>
                            <div className="rounded-xl border border-graphite-700/70 bg-graphite-900 px-3 py-3 sm:col-span-2">
                                <div className="text-xs uppercase tracking-[0.24em] text-secondary">Package Folder</div>
                                <div className="mt-1 break-all font-mono text-xs">{deployInspect.deployment.package_dir || 'Not available'}</div>
                            </div>
                            {deployInspect.deployment.readme_path && (
                                <div className="rounded-xl border border-graphite-700/70 bg-graphite-900 px-3 py-3 sm:col-span-2">
                                    <div className="text-xs uppercase tracking-[0.24em] text-secondary">Readme</div>
                                    <div className="mt-1 break-all font-mono text-xs">{deployInspect.deployment.readme_path}</div>
                                </div>
                            )}
                            <div className="rounded-xl border border-graphite-700/70 bg-graphite-900 px-3 py-3 sm:col-span-2">
                                <div className="text-xs uppercase tracking-[0.24em] text-secondary">Created At</div>
                                <div className="mt-1 text-sm">{formatDate(deployInspect.deployment.created_at)}</div>
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
