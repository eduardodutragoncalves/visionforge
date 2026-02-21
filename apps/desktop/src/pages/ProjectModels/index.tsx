import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react';
import { useParams } from 'react-router-dom';
import type { LucideIcon } from 'lucide-react';
import {
    Activity,
    Box,
    ClipboardList,
    CheckCircle2,
    Clock3,
    Cpu,
    ExternalLink,
    FileText,
    HardDriveDownload,
    LayoutGrid,
    Loader2,
    Monitor,
    Rocket,
    Server,
    Target,
    Trash2,
    XCircle
} from 'lucide-react';
import { Card } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { Input } from '../../components/ui/Input';
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
    package_zip_path?: string;
    readme_path?: string;
    created_at?: string;
    status?: string;
}

interface ValidationSummary {
    total_images?: number;
    tested_images?: number;
    untested_images?: number;
    coverage_pct?: number;
    correct?: number;
    incorrect?: number;
    pending?: number;
    labeled?: number;
    accuracy_pct?: number | null;
    last_tested_at?: string | null;
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
    validation?: ValidationSummary;
}

type DeployState = {
    targetId: string;
    status: 'queued' | 'running';
    message?: string;
    stage?: string;
    progress?: number;
    startedAt: number;
    runId?: string;
    cancelRequested?: boolean;
};

type DeploymentInspect = {
    variant: Variant;
    deployment: DeploymentInfo;
};

type DeploymentRecord = {
    key: string;
    variant: Variant;
    deployment: DeploymentInfo;
};

type ReadmeInspect = {
    title: string;
    targetLabel: string;
    path: string;
    content: string;
};

interface SummaryMetricCardProps {
    title: string;
    value: string;
    hint: string;
    icon: LucideIcon;
    tone?: 'default' | 'success' | 'warning' | 'running';
}

type ModelValidationFilter = 'all' | 'validated' | 'not_validated';
type ModelDeploymentFilter = 'all' | 'deployed' | 'not_deployed' | 'deploying';
type ModelSortBy =
    | 'newest'
    | 'oldest'
    | 'train_accuracy_desc'
    | 'validation_accuracy_desc'
    | 'validation_coverage_desc'
    | 'deployments_desc'
    | 'name_asc';

type DeploymentSortBy = 'newest' | 'oldest' | 'variant_asc' | 'target_asc';

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

const formatPercent = (value?: number | null, decimals = 1) => {
    if (typeof value !== 'number' || Number.isNaN(value)) return '—';
    return `${value.toFixed(decimals)}%`;
};

const toAccuracyPercent = (value?: number | null) => {
    if (typeof value !== 'number' || Number.isNaN(value)) return null;
    if (value <= 1) return value * 100;
    return value;
};

const formatDate = (value?: string) => {
    if (!value) return 'Unknown date';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleString();
};

const formatAccuracyBadge = (value?: number | null) => {
    const percent = toAccuracyPercent(value);
    if (percent === null) return '—';
    return `${percent.toFixed(1)}%`;
};

const normalizedValidation = (validation?: ValidationSummary) => {
    const totalImages = Math.max(0, Number(validation?.total_images || 0));
    const testedImages = Math.max(0, Number(validation?.tested_images || 0));
    const correct = Math.max(0, Number(validation?.correct || 0));
    const incorrect = Math.max(0, Number(validation?.incorrect || 0));
    const pending = Math.max(0, Number(validation?.pending || 0));
    const labeled = Math.max(0, Number(validation?.labeled || correct + incorrect));
    const coverage = typeof validation?.coverage_pct === 'number'
        ? validation.coverage_pct
        : (totalImages > 0 ? (testedImages / totalImages) * 100 : 0);
    return {
        totalImages,
        testedImages,
        correct,
        incorrect,
        pending,
        labeled,
        coverage: Math.max(0, Math.min(100, coverage)),
        accuracyPct: typeof validation?.accuracy_pct === 'number' ? validation.accuracy_pct : null,
        lastTestedAt: validation?.last_tested_at || null
    };
};

const renderInlineMarkdown = (text: string, keyPrefix: string): ReactNode[] => {
    const parts = text.split(/(`[^`]+`)/g);
    return parts.map((part, idx) => {
        const key = `${keyPrefix}-${idx}`;
        if (part.startsWith('`') && part.endsWith('`')) {
            return (
                <code key={key} className="rounded border border-graphite-700/80 bg-graphite-900 px-1.5 py-0.5 font-mono text-[11px] text-accent-primary">
                    {part.slice(1, -1)}
                </code>
            );
        }
        return <span key={key}>{part}</span>;
    });
};

const renderReadmeMarkdown = (markdown: string): ReactNode[] => {
    const lines = markdown.replace(/\r\n/g, '\n').split('\n');
    const nodes: ReactNode[] = [];
    let paragraph: string[] = [];
    let listItems: string[] = [];
    let listType: 'ul' | 'ol' | null = null;
    let codeBlock: string[] = [];
    let inCodeBlock = false;
    let seq = 0;

    const flushParagraph = () => {
        if (paragraph.length === 0) return;
        const text = paragraph.join(' ').trim();
        if (!text) {
            paragraph = [];
            return;
        }
        seq += 1;
        nodes.push(
            <p key={`p-${seq}`} className="text-sm leading-6 text-bone">
                {renderInlineMarkdown(text, `p-${seq}`)}
            </p>
        );
        paragraph = [];
    };

    const flushList = () => {
        if (listItems.length === 0 || !listType) return;
        seq += 1;
        if (listType === 'ul') {
            nodes.push(
                <ul key={`ul-${seq}`} className="list-disc space-y-1 pl-5 text-sm text-bone">
                    {listItems.map((item, idx) => (
                        <li key={`ul-${seq}-${idx}`}>{renderInlineMarkdown(item, `ul-${seq}-${idx}`)}</li>
                    ))}
                </ul>
            );
        } else {
            nodes.push(
                <ol key={`ol-${seq}`} className="list-decimal space-y-1 pl-5 text-sm text-bone">
                    {listItems.map((item, idx) => (
                        <li key={`ol-${seq}-${idx}`}>{renderInlineMarkdown(item, `ol-${seq}-${idx}`)}</li>
                    ))}
                </ol>
            );
        }
        listItems = [];
        listType = null;
    };

    const flushCode = () => {
        if (codeBlock.length === 0) return;
        seq += 1;
        nodes.push(
            <pre key={`code-${seq}`} className="overflow-x-auto rounded-lg border border-graphite-700/70 bg-graphite-950 p-3 font-mono text-xs leading-5 text-accent-primary">
                <code>{codeBlock.join('\n')}</code>
            </pre>
        );
        codeBlock = [];
    };

    for (const rawLine of lines) {
        const line = rawLine ?? '';
        const trimmed = line.trim();

        if (trimmed.startsWith('```')) {
            flushParagraph();
            flushList();
            if (inCodeBlock) {
                flushCode();
                inCodeBlock = false;
            } else {
                inCodeBlock = true;
                codeBlock = [];
            }
            continue;
        }

        if (inCodeBlock) {
            codeBlock.push(line);
            continue;
        }

        if (!trimmed) {
            flushParagraph();
            flushList();
            continue;
        }

        const headingMatch = /^#{1,3}\s+(.+)$/.exec(trimmed);
        if (headingMatch) {
            flushParagraph();
            flushList();
            seq += 1;
            const level = trimmed.match(/^#+/)?.[0].length || 1;
            const text = headingMatch[1].trim();
            if (level === 1) {
                nodes.push(
                    <h1 key={`h1-${seq}`} className="text-2xl font-semibold text-bone">
                        {renderInlineMarkdown(text, `h1-${seq}`)}
                    </h1>
                );
            } else if (level === 2) {
                nodes.push(
                    <h2 key={`h2-${seq}`} className="text-lg font-semibold uppercase tracking-[0.18em] text-accent-primary">
                        {renderInlineMarkdown(text, `h2-${seq}`)}
                    </h2>
                );
            } else {
                nodes.push(
                    <h3 key={`h3-${seq}`} className="text-base font-semibold text-bone">
                        {renderInlineMarkdown(text, `h3-${seq}`)}
                    </h3>
                );
            }
            continue;
        }

        const orderedMatch = /^\d+\.\s+(.+)$/.exec(trimmed);
        if (orderedMatch) {
            flushParagraph();
            if (listType && listType !== 'ol') flushList();
            listType = 'ol';
            listItems.push(orderedMatch[1]);
            continue;
        }

        const unorderedMatch = /^[-*]\s+(.+)$/.exec(trimmed);
        if (unorderedMatch) {
            flushParagraph();
            if (listType && listType !== 'ul') flushList();
            listType = 'ul';
            listItems.push(unorderedMatch[1]);
            continue;
        }

        if (listType) {
            flushList();
        }
        paragraph.push(trimmed);
    }

    flushParagraph();
    flushList();
    flushCode();

    if (nodes.length === 0) {
        return [<p key="empty-readme" className="text-sm text-secondary">Empty README.</p>];
    }
    return nodes;
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
    const [readmeInspect, setReadmeInspect] = useState<ReadmeInspect | null>(null);
    const [selectedTargetId, setSelectedTargetId] = useState<string>('edge_jetson');
    const [deployingByVariant, setDeployingByVariant] = useState<Record<string, DeployState>>({});
    const [deployNotice, setDeployNotice] = useState<string | null>(null);
    const [readmeLoading, setReadmeLoading] = useState(false);
    const [nowTs, setNowTs] = useState<number>(Date.now());
    const [modelSearchQuery, setModelSearchQuery] = useState('');
    const [modelValidationFilter, setModelValidationFilter] = useState<ModelValidationFilter>('all');
    const [modelDeploymentFilter, setModelDeploymentFilter] = useState<ModelDeploymentFilter>('all');
    const [modelSortBy, setModelSortBy] = useState<ModelSortBy>('newest');
    const [deploymentSearchQuery, setDeploymentSearchQuery] = useState('');
    const [deploymentTargetFilter, setDeploymentTargetFilter] = useState<string>('all');
    const [deploymentSortBy, setDeploymentSortBy] = useState<DeploymentSortBy>('newest');
    const [deletingDeploymentKey, setDeletingDeploymentKey] = useState<string | null>(null);

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
                    startedAt: prev[variantId]?.startedAt || Date.now(),
                    runId: detail.run_id || prev[variantId]?.runId,
                    cancelRequested: false
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
                        runId: detail.run_id || current.runId,
                        cancelRequested: detail.cancel_requested ? true : current.cancelRequested
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
            setDeployNotice(`Deployment completed for ${variantId} (${targetLabel(detail.target_id)}).`);
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
            if (detail.cancelled) {
                setDeployNotice(`Deployment cancelled for ${variantId || 'job'} (${targetLabel(detail.target_id)}).`);
                return;
            }
            setDeployNotice(`Deployment failed: ${detail.error || 'Unknown error'}`);
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
                startedAt: Date.now(),
                runId: undefined,
                cancelRequested: false
            }
        }));
        setDeployModalVariant(null);

        const res = await fetch(`${ENGINE_HTTP_URL}/projects/${projectId}/variants/${variantId}/deploy`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ target_id: selectedTargetId })
        });

        if (res.ok) {
            try {
                const payload = await res.json();
                if (payload?.job_id) {
                    setDeployingByVariant((prev) => {
                        const current = prev[variantId];
                        if (!current) return prev;
                        return {
                            ...prev,
                            [variantId]: {
                                ...current,
                                runId: String(payload.job_id)
                            }
                        };
                    });
                }
            } catch {
                // ignore
            }
            return;
        }

        if (!res.ok) {
            setDeployingByVariant((prev) => {
                const next = { ...prev };
                delete next[variantId];
                return next;
            });
            const text = await res.text();
            setDeployNotice(`Failed to queue deployment: ${text}`);
        }
    };

    const cancelDeploy = useCallback(async (variantId: string) => {
        if (!projectId) return;
        const current = deployingByVariant[variantId];
        if (!current) return;
        if (current.cancelRequested) return;
        if (!current.runId) {
            setDeployNotice('Cancellation is unavailable until the deployment job ID is assigned.');
            return;
        }

        setDeployingByVariant((prev) => {
            const state = prev[variantId];
            if (!state) return prev;
            return {
                ...prev,
                [variantId]: {
                    ...state,
                    cancelRequested: true,
                    message: 'Cancellation requested...'
                }
            };
        });

        try {
            const res = await fetch(`${ENGINE_HTTP_URL}/projects/${projectId}/deployments/${current.runId}/cancel`, {
                method: 'POST'
            });
            if (!res.ok) {
                const text = await res.text();
                throw new Error(text || 'Failed to cancel deployment');
            }
            const payload = await res.json();
            const status = String(payload?.status || '');
            if (status === 'cancelled_queued') {
                setDeployingByVariant((prev) => {
                    const next = { ...prev };
                    delete next[variantId];
                    return next;
                });
                setDeployNotice(`Deployment cancelled for ${variantId} before execution.`);
                return;
            }
            if (status === 'cancel_requested') {
                setDeployingByVariant((prev) => {
                    const next = { ...prev };
                    delete next[variantId];
                    return next;
                });
                setDeployNotice(`Cancellation requested for ${variantId}. Cleaning generated deployment files...`);
                return;
            }
            setDeployNotice(`Cancellation requested for ${variantId}.`);
        } catch (error: any) {
            setDeployingByVariant((prev) => {
                const state = prev[variantId];
                if (!state) return prev;
                return {
                    ...prev,
                    [variantId]: {
                        ...state,
                        cancelRequested: false
                    }
                };
            });
            setDeployNotice(`Failed to cancel deployment: ${error?.message || 'unknown error'}`);
        }
    }, [deployingByVariant, projectId]);

    const handleDeleteDeployment = useCallback(async (variant: Variant, deployment: DeploymentInfo) => {
        if (!projectId) return;
        const deploymentId = String(deployment.id || '').trim();
        if (!deploymentId) {
            setDeployNotice('Cannot delete deployment: missing deployment id.');
            return;
        }
        const key = `${variant.id}::${deploymentId}::${deployment.package_dir || ''}`;
        const confirmed = window.confirm(
            `Delete deployment ${deploymentId} for ${variant.name} (${targetLabel(deployment.target_id)})?\n\n` +
            'This removes the package folder and generated zip.'
        );
        if (!confirmed) return;

        setDeletingDeploymentKey(key);
        try {
            const res = await fetch(
                `${ENGINE_HTTP_URL}/projects/${projectId}/variants/${variant.id}/deployments/${deploymentId}`,
                { method: 'DELETE' }
            );
            if (!res.ok) {
                const text = await res.text();
                throw new Error(text || 'Failed to delete deployment');
            }
            if (deployInspect && deployInspect.variant.id === variant.id && String(deployInspect.deployment.id || '') === deploymentId) {
                setDeployInspect(null);
            }
            setDeployNotice(`Deployment ${deploymentId} deleted successfully.`);
            await fetchVariants();
        } catch (error: any) {
            setDeployNotice(`Failed to delete deployment: ${error?.message || 'unknown error'}`);
        } finally {
            setDeletingDeploymentKey(null);
        }
    }, [deployInspect, fetchVariants, projectId]);

    const activeDeployEntries = useMemo(
        () => Object.entries(deployingByVariant),
        [deployingByVariant]
    );

    const deploymentRecords = useMemo<DeploymentRecord[]>(() => {
        const rows: DeploymentRecord[] = [];
        variants.forEach((variant) => {
            (variant.deployments || []).forEach((deployment, index) => {
                const key = `${variant.id}::${deployment.id || `idx_${index}`}::${deployment.package_dir || ''}`;
                rows.push({ key, variant, deployment });
            });
        });
        return rows;
    }, [variants]);

    const filteredDeploymentRecords = useMemo(() => {
        const query = deploymentSearchQuery.trim().toLowerCase();
        const parseTimestamp = (value?: string) => {
            const ts = Date.parse(String(value || ''));
            return Number.isNaN(ts) ? 0 : ts;
        };

        const rows = deploymentRecords.filter((record) => {
            const targetId = String(record.deployment.target_id || '').trim();
            if (deploymentTargetFilter !== 'all' && targetId !== deploymentTargetFilter) {
                return false;
            }
            if (!query) return true;
            const searchable = [
                record.variant.name,
                record.variant.id,
                record.deployment.id,
                record.deployment.target_id,
                record.deployment.target_label,
                record.deployment.format,
                record.deployment.package_dir,
                record.deployment.package_zip_path
            ].join(' ').toLowerCase();
            return searchable.includes(query);
        });

        rows.sort((a, b) => {
            if (deploymentSortBy === 'newest') {
                return parseTimestamp(b.deployment.created_at) - parseTimestamp(a.deployment.created_at);
            }
            if (deploymentSortBy === 'oldest') {
                return parseTimestamp(a.deployment.created_at) - parseTimestamp(b.deployment.created_at);
            }
            if (deploymentSortBy === 'variant_asc') {
                return String(a.variant.name || a.variant.id).localeCompare(String(b.variant.name || b.variant.id));
            }
            return targetLabel(a.deployment.target_id).localeCompare(targetLabel(b.deployment.target_id));
        });

        return rows;
    }, [deploymentRecords, deploymentSearchQuery, deploymentSortBy, deploymentTargetFilter]);

    const filteredVariants = useMemo(() => {
        const query = modelSearchQuery.trim().toLowerCase();
        const parseTimestamp = (value?: string) => {
            const ts = Date.parse(String(value || ''));
            return Number.isNaN(ts) ? 0 : ts;
        };

        const filtered = variants.filter((variant) => {
            const validation = normalizedValidation(variant.validation);
            const hasValidation = validation.testedImages > 0;
            const hasDeployment = (variant.deployments?.length || 0) > 0;
            const isDeploying = Boolean(deployingByVariant[variant.id]);

            if (modelValidationFilter === 'validated' && !hasValidation) return false;
            if (modelValidationFilter === 'not_validated' && hasValidation) return false;

            if (modelDeploymentFilter === 'deployed' && !hasDeployment) return false;
            if (modelDeploymentFilter === 'not_deployed' && hasDeployment) return false;
            if (modelDeploymentFilter === 'deploying' && !isDeploying) return false;

            if (query) {
                const searchable = [
                    variant.id,
                    variant.name,
                    variant.base_model,
                    variant.run_id
                ].join(' ').toLowerCase();
                if (!searchable.includes(query)) return false;
            }

            return true;
        });

        filtered.sort((a, b) => {
            if (modelSortBy === 'newest') return parseTimestamp(b.created_at) - parseTimestamp(a.created_at);
            if (modelSortBy === 'oldest') return parseTimestamp(a.created_at) - parseTimestamp(b.created_at);
            if (modelSortBy === 'name_asc') return String(a.name || a.id).localeCompare(String(b.name || b.id));
            if (modelSortBy === 'deployments_desc') return (b.deployments?.length || 0) - (a.deployments?.length || 0);
            if (modelSortBy === 'train_accuracy_desc') {
                const aAcc = toAccuracyPercent(a.metrics?.accuracy) ?? -Infinity;
                const bAcc = toAccuracyPercent(b.metrics?.accuracy) ?? -Infinity;
                return bAcc - aAcc;
            }
            if (modelSortBy === 'validation_accuracy_desc') {
                const aAcc = normalizedValidation(a.validation).accuracyPct ?? -Infinity;
                const bAcc = normalizedValidation(b.validation).accuracyPct ?? -Infinity;
                return bAcc - aAcc;
            }
            const aCoverage = normalizedValidation(a.validation).coverage;
            const bCoverage = normalizedValidation(b.validation).coverage;
            return bCoverage - aCoverage;
        });

        return filtered;
    }, [deployingByVariant, modelDeploymentFilter, modelSearchQuery, modelSortBy, modelValidationFilter, variants]);

    const totalDeployments = useMemo(
        () => variants.reduce((count, variant) => count + (variant.deployments?.length || 0), 0),
        [variants]
    );

    const modelsWithValidation = useMemo(
        () => variants.filter((variant) => normalizedValidation(variant.validation).testedImages > 0).length,
        [variants]
    );

    const validationLabeledTotal = useMemo(
        () => variants.reduce((count, variant) => count + normalizedValidation(variant.validation).labeled, 0),
        [variants]
    );

    const averageValidationAccuracy = useMemo(() => {
        const entries = variants
            .map((variant) => normalizedValidation(variant.validation).accuracyPct)
            .filter((accuracy): accuracy is number => typeof accuracy === 'number' && !Number.isNaN(accuracy));
        if (entries.length === 0) return null;
        return entries.reduce((sum, value) => sum + value, 0) / entries.length;
    }, [variants]);

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

    const openReadmePreview = useCallback(async (variant: Variant, deployment: DeploymentInfo) => {
        const readmePath = deployment.readme_path;
        if (!readmePath) {
            setDeployNotice('README not found for this deployment.');
            return;
        }
        const reader = window.ipcRenderer?.readTextFile;
        if (!reader) {
            setDeployNotice('Local file access is unavailable in this session.');
            return;
        }
        setReadmeLoading(true);
        try {
            const result = await reader(readmePath);
            if (!result?.ok || typeof result.content !== 'string') {
                setDeployNotice(`Could not open README: ${result?.message || 'unknown error'}`);
                return;
            }
            setReadmeInspect({
                title: variant.name,
                targetLabel: targetLabel(deployment.target_id),
                path: readmePath,
                content: result.content
            });
        } finally {
            setReadmeLoading(false);
        }
    }, []);

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

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-5">
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
                <SummaryMetricCard
                    title="Validation"
                    value={`${modelsWithValidation}/${variants.length}`}
                    hint={`Accuracy avg ${formatPercent(averageValidationAccuracy)} · Labels ${validationLabeledTotal}`}
                    icon={CheckCircle2}
                    tone={modelsWithValidation > 0 ? 'success' : 'default'}
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
                                    <div className="mt-2 flex justify-end">
                                        <Button
                                            size="sm"
                                            variant="ghost"
                                            onClick={() => cancelDeploy(variantId)}
                                            disabled={state.cancelRequested}
                                            className="text-error hover:text-error"
                                        >
                                            {state.cancelRequested ? 'Cancelling...' : 'Cancel Deploy'}
                                        </Button>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </Card>
            )}

            <Card className="border border-graphite-700/70 bg-graphite-900/35">
                <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
                    <Input
                        label="Search"
                        placeholder="Model name, ID, base model"
                        value={modelSearchQuery}
                        onChange={(e) => setModelSearchQuery(e.target.value)}
                    />
                    <Select
                        label="Validation"
                        value={modelValidationFilter}
                        onChange={(e) => setModelValidationFilter(e.target.value as ModelValidationFilter)}
                        options={[
                            { value: 'all', label: 'All models' },
                            { value: 'validated', label: 'Validated only' },
                            { value: 'not_validated', label: 'Not validated' }
                        ]}
                    />
                    <Select
                        label="Deployment"
                        value={modelDeploymentFilter}
                        onChange={(e) => setModelDeploymentFilter(e.target.value as ModelDeploymentFilter)}
                        options={[
                            { value: 'all', label: 'All states' },
                            { value: 'deployed', label: 'Deployed' },
                            { value: 'not_deployed', label: 'Not deployed' },
                            { value: 'deploying', label: 'Deploying now' }
                        ]}
                    />
                    <Select
                        label="Sort"
                        value={modelSortBy}
                        onChange={(e) => setModelSortBy(e.target.value as ModelSortBy)}
                        options={[
                            { value: 'newest', label: 'Newest first' },
                            { value: 'oldest', label: 'Oldest first' },
                            { value: 'train_accuracy_desc', label: 'Train accuracy (desc)' },
                            { value: 'validation_accuracy_desc', label: 'Validation accuracy (desc)' },
                            { value: 'validation_coverage_desc', label: 'Validation coverage (desc)' },
                            { value: 'deployments_desc', label: 'Deployments count (desc)' },
                            { value: 'name_asc', label: 'Name (A-Z)' }
                        ]}
                    />
                </div>
                <div className="mt-3 text-xs uppercase tracking-[0.24em] text-secondary">
                    Showing {filteredVariants.length} of {variants.length} models
                </div>
            </Card>

            <div className="grid grid-cols-1 gap-3 xl:grid-cols-2 2xl:grid-cols-3">
                {variants.length === 0 && (
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

                {variants.length > 0 && filteredVariants.length === 0 && (
                    <Card className="col-span-full border border-graphite-700/70 bg-graphite-900/40 py-12">
                        <div className="mx-auto max-w-md text-center text-secondary">
                            <Activity size={20} className="mx-auto mb-3 text-haze" />
                            <div className="mb-2 text-sm uppercase tracking-[0.3em]">No matches</div>
                            <div className="text-sm">No model matches the selected filters and sorting options.</div>
                        </div>
                    </Card>
                )}

                {filteredVariants.map((variant) => {
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
                    const validation = normalizedValidation(variant.validation);
                    const hasValidation = validation.testedImages > 0;

                    return (
                        <Card
                            key={variant.id}
                            className="flex h-full flex-col border border-graphite-700/70 bg-gradient-to-b from-graphite-900/80 to-graphite-900/50 p-4 transition-all duration-150 hover:border-accent-primary/40"
                        >
                            <div className="flex items-start justify-between gap-2">
                                <div className="min-w-0">
                                    <h3 className="truncate text-lg font-semibold">{variant.name}</h3>
                                    <div className="mt-0.5 truncate font-mono text-[11px] text-secondary">{variant.id}</div>
                                </div>
                                <StatusPill status={status} label={statusLabel} pulsing={Boolean(activeDeploy)} />
                            </div>

                            <div className="mt-3 grid grid-cols-2 gap-2">
                                <div className="rounded-lg border border-accent-primary/30 bg-accent-primary/10 px-2 py-2">
                                    <div className="text-[10px] uppercase tracking-[0.2em] text-secondary">Train accuracy</div>
                                    <div className="mt-1 text-lg font-mono text-accent-primary">{formatAccuracyBadge(variant.metrics.accuracy)}</div>
                                </div>
                                <div className={`rounded-lg border px-2 py-2 ${hasValidation ? 'border-accent-success/35 bg-accent-success/10' : 'border-graphite-700/70 bg-graphite-900/70'}`}>
                                    <div className="text-[10px] uppercase tracking-[0.2em] text-secondary">Validation accuracy</div>
                                    <div className="mt-1 text-lg font-mono text-accent-success">{formatPercent(validation.accuracyPct)}</div>
                                </div>
                            </div>

                            <div className="mt-2.5 grid grid-cols-2 gap-1.5 text-[10px] uppercase tracking-[0.16em] text-secondary">
                                <div className="rounded border border-graphite-700/70 bg-graphite-900/70 px-2 py-1.5">
                                    mAP50 <strong className="text-bone">{formatMetric(variant.metrics.final_map)}</strong>
                                </div>
                                <div className="rounded border border-graphite-700/70 bg-graphite-900/70 px-2 py-1.5">
                                    Loss <strong className="text-bone">{formatMetric(variant.metrics.loss)}</strong>
                                </div>
                                <div className="rounded border border-graphite-700/70 bg-graphite-900/70 px-2 py-1.5 col-span-2">
                                    Coverage <strong className="text-bone">{formatPercent(validation.coverage)}</strong>
                                </div>
                            </div>

                            <div className="mt-2.5 flex flex-wrap gap-1.5 text-[10px] uppercase tracking-[0.16em] text-secondary">
                                <span className="rounded border border-graphite-700/70 bg-graphite-900/70 px-2 py-1.5">
                                    Base model <strong className="text-bone">{variant.base_model}</strong>
                                </span>
                                <span className="rounded border border-graphite-700/70 bg-graphite-900/70 px-2 py-1.5">
                                    Created <strong className="text-bone">{formatDate(variant.created_at)}</strong>
                                </span>
                            </div>

                            <div className="mt-2.5 rounded-lg border border-graphite-700/70 bg-graphite-900/55 px-2.5 py-2">
                                <div className="text-[10px] uppercase tracking-[0.25em] text-secondary">Deployment state</div>
                                <div className="mt-1 text-xs">{statusHint}</div>
                                {activeDeploy && (
                                    <div className="mt-2 h-1.5 overflow-hidden rounded bg-graphite-700">
                                        <div
                                            className="h-full bg-accent-warning transition-all duration-500"
                                            style={{ width: `${Math.max(5, Math.min(97, activeDeploy.progress ?? 0))}%` }}
                                        />
                                    </div>
                                )}
                            </div>

                            <div className={`mt-2.5 rounded-lg border px-2.5 py-2 ${hasValidation ? 'border-accent-success/35 bg-accent-success/5' : 'border-graphite-700/70 bg-graphite-900/55'}`}>
                                <div className="text-[10px] uppercase tracking-[0.25em] text-secondary">Validation details</div>
                                {hasValidation ? (
                                    <>
                                        <div className="mt-1 text-[11px] text-secondary">
                                            {validation.testedImages}/{validation.totalImages} images tested
                                        </div>
                                        <div className="mt-1.5 h-1.5 overflow-hidden rounded bg-graphite-700">
                                            <div className="h-full bg-accent-success transition-all duration-300" style={{ width: `${validation.coverage}%` }} />
                                        </div>
                                        <div className="mt-1.5 flex flex-wrap gap-1.5 text-[10px] uppercase tracking-[0.14em] text-secondary">
                                            <span className="rounded border border-accent-success/35 bg-accent-success/10 px-1.5 py-1">
                                                Correct <strong className="text-bone">{validation.correct}</strong>
                                            </span>
                                            <span className="rounded border border-accent-error/35 bg-accent-error/10 px-1.5 py-1">
                                                Wrong <strong className="text-bone">{validation.incorrect}</strong>
                                            </span>
                                            <span className="rounded border border-graphite-700/70 bg-graphite-900 px-1.5 py-1">
                                                Pending <strong className="text-bone">{validation.pending}</strong>
                                            </span>
                                        </div>
                                        <div className="mt-1.5 text-[11px] text-secondary">
                                            Last test {validation.lastTestedAt ? formatDate(validation.lastTestedAt) : '—'}
                                        </div>
                                    </>
                                ) : (
                                    <div className="mt-1 text-xs text-secondary">Model not validated yet on inference history images.</div>
                                )}
                            </div>

                            <div className="mt-3">
                                <div className="mb-1.5 text-[10px] uppercase tracking-[0.25em] text-secondary">Deploy targets</div>
                                {deployedTargets.length === 0 ? (
                                    <div className="rounded-lg border border-dashed border-graphite-700/80 px-2.5 py-2 text-[11px] text-secondary">
                                        No target exported yet. Click <strong>Deploy</strong> to generate the first package.
                                    </div>
                                ) : (
                                    <div className="flex flex-wrap gap-1.5">
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

                            <div className="mt-3 grid grid-cols-2 gap-2">
                                <Button size="sm" variant="secondary">Test</Button>
                                {activeDeploy ? (
                                    <Button
                                        size="sm"
                                        variant="ghost"
                                        onClick={() => cancelDeploy(variant.id)}
                                        disabled={activeDeploy.cancelRequested}
                                        className="text-error hover:text-error"
                                    >
                                        {activeDeploy.cancelRequested ? 'Cancelling...' : 'Cancel'}
                                    </Button>
                                ) : (
                                    <Button
                                        size="sm"
                                        variant="primary"
                                        onClick={() => {
                                            setSelectedTargetId('edge_jetson');
                                            setDeployModalVariant(variant);
                                        }}
                                    >
                                        <HardDriveDownload size={14} className="mr-1" />
                                        Deploy
                                    </Button>
                                )}
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
                            {deployInspect.deployment.package_zip_path && (
                                <div className="rounded-xl border border-graphite-700/70 bg-graphite-900 px-3 py-3 sm:col-span-2">
                                    <div className="text-xs uppercase tracking-[0.24em] text-secondary">Package Zip</div>
                                    <div className="mt-1 break-all font-mono text-xs">{deployInspect.deployment.package_zip_path}</div>
                                </div>
                            )}
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
                                variant="secondary"
                                onClick={() => openReadmePreview(deployInspect.variant, deployInspect.deployment)}
                                disabled={readmeLoading || !deployInspect.deployment.readme_path}
                            >
                                {readmeLoading ? (
                                    <Loader2 size={14} className="mr-1 animate-spin" />
                                ) : (
                                    <FileText size={14} className="mr-1" />
                                )}
                                Open Readme
                            </Button>
                            <Button
                                variant="ghost"
                                onClick={async () => {
                                    const artifact = deployInspect.deployment.artifact_path;
                                    if (!artifact) return;
                                    const result = await window.ipcRenderer?.revealPath?.(artifact);
                                    if (result?.ok === false) {
                                        setDeployNotice(`Could not open artifact: ${result.message || 'unknown error'}`);
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
                                        setDeployNotice(`Could not open folder: ${result.message || 'unknown error'}`);
                                    }
                                }}
                            >
                                <ExternalLink size={14} className="mr-1" />
                                Open Package Folder
                            </Button>
                            <Button
                                variant="ghost"
                                onClick={async () => {
                                    const zip = deployInspect.deployment.package_zip_path;
                                    if (!zip) return;
                                    const result = await window.ipcRenderer?.revealPath?.(zip);
                                    if (result?.ok === false) {
                                        setDeployNotice(`Could not open zip: ${result.message || 'unknown error'}`);
                                    }
                                }}
                                disabled={!deployInspect.deployment.package_zip_path}
                            >
                                <ExternalLink size={14} className="mr-1" />
                                Reveal Zip
                            </Button>
                        </div>
                    </div>
                </div>
            )}

            {readmeInspect && (
                <div className="modal-backdrop">
                    <div className="modal-card max-w-4xl">
                        <div className="modal-header mb-4">
                            <div>
                                <h3 className="text-2xl font-semibold">Deployment Guide</h3>
                                <p className="mt-1 text-sm text-secondary">
                                    {readmeInspect.title} · {readmeInspect.targetLabel}
                                </p>
                                <p className="mt-1 break-all font-mono text-[11px] text-secondary">{readmeInspect.path}</p>
                            </div>
                            <Button size="sm" variant="ghost" onClick={() => setReadmeInspect(null)}>
                                <XCircle size={14} className="mr-1" />
                                Close
                            </Button>
                        </div>

                        <Card className="max-h-[70vh] overflow-y-auto border border-graphite-700/70 bg-graphite-900/70">
                            <div className="mb-3 flex items-center gap-2 text-xs uppercase tracking-[0.24em] text-secondary">
                                <ClipboardList size={13} />
                                Step-by-step instructions
                            </div>
                            <div className="space-y-3">
                                {renderReadmeMarkdown(readmeInspect.content)}
                            </div>
                        </Card>

                        <div className="mt-4 flex justify-end gap-2">
                            <Button
                                variant="ghost"
                                onClick={async () => {
                                    const result = await window.ipcRenderer?.revealPath?.(readmeInspect.path);
                                    if (result?.ok === false) {
                                        setDeployNotice(`Could not open README: ${result.message || 'unknown error'}`);
                                    }
                                }}
                            >
                                <ExternalLink size={14} className="mr-1" />
                                Reveal File
                            </Button>
                            <Button
                                variant="secondary"
                                onClick={() => setReadmeInspect(null)}
                            >
                                Close Guide
                            </Button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};
