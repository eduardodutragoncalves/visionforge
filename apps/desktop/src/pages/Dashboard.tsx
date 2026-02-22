import { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    Activity,
    Boxes,
    CheckCircle2,
    Cpu,
    FolderOpen,
    LayoutGrid,
    PlayCircle,
    RefreshCcw
} from 'lucide-react';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { StatusPill } from '../components/ui/StatusPill';
import { ENGINE_HTTP_URL } from '../lib/engine';
import { useEngineStore } from '../store/engineStore';
import { useTrainingStore } from '../store/trainingStore';

interface ProjectSummary {
    id: string;
    name: string;
    analysis_type?: 'classification' | 'detection';
    created_at: string;
    updated_at: string;
    images_total?: number;
    labeled_images?: number;
    labels_total?: number;
    classes_total?: number;
    runs_total?: number;
    last_run_at?: string | null;
}

interface CapabilitiesResponse {
    gpu_available: boolean;
    gpu_name: string;
    frameworks: string[];
    version: string;
}

const toLocaleDate = (value?: string | null) => {
    if (!value) return '—';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleString();
};

const formatPercent = (value: number) => `${Math.max(0, Math.min(100, value)).toFixed(1)}%`;

const formatNumber = (value: number) => new Intl.NumberFormat().format(Math.max(0, value));

const classifyRunStatus = (status?: string) => {
    const normalized = String(status || '').toLowerCase();
    if (normalized === 'training' || normalized === 'starting') return 'running';
    if (normalized === 'completed') return 'success';
    if (normalized === 'failed' || normalized === 'error') return 'error';
    return 'idle';
};

const DASHBOARD_BOOT_TIMEOUT_MS = 12000;

const fetchWithTimeout = async (url: string, timeoutMs = DASHBOARD_BOOT_TIMEOUT_MS) => {
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), timeoutMs);
    try {
        return await fetch(url, { signal: controller.signal });
    } finally {
        window.clearTimeout(timer);
    }
};

export const Dashboard = () => {
    const navigate = useNavigate();
    const { isConnected, status, gpu, gpus, setDashboardReady } = useEngineStore((state) => ({
        isConnected: state.isConnected,
        status: state.status,
        gpu: state.gpu,
        gpus: state.gpus,
        setDashboardReady: state.setDashboardReady
    }));
    const trainingRuns = useTrainingStore((state) => Object.values(state.runs));

    const [projects, setProjects] = useState<ProjectSummary[]>([]);
    const [capabilities, setCapabilities] = useState<CapabilitiesResponse | null>(null);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [lastSyncAt, setLastSyncAt] = useState<string | null>(null);

    const loadDashboard = useCallback(async (background = false) => {
        if (background) {
            setRefreshing(true);
        } else {
            setLoading(true);
        }
        setError(null);
        try {
            const [projectsRes, capabilitiesRes] = await Promise.all([
                fetchWithTimeout(`${ENGINE_HTTP_URL}/projects`),
                fetchWithTimeout(`${ENGINE_HTTP_URL}/capabilities`)
            ]);

            if (!projectsRes.ok) {
                throw new Error('Failed to load project overview.');
            }
            const projectsData = await projectsRes.json();
            setProjects(Array.isArray(projectsData) ? projectsData : []);

            if (capabilitiesRes.ok) {
                const capabilitiesData = await capabilitiesRes.json();
                setCapabilities(capabilitiesData);
            } else {
                setCapabilities(null);
            }
            setLastSyncAt(new Date().toISOString());
        } catch (err) {
            if (err instanceof DOMException && err.name === 'AbortError') {
                setError('Dashboard request timed out. Check backend connectivity and try refresh.');
            } else {
                setError(err instanceof Error ? err.message : 'Failed to load dashboard information.');
            }
        } finally {
            if (!background) {
                setLoading(false);
                // First dashboard load finished (success or error), release splash.
                setDashboardReady(true);
            }
            setRefreshing(false);
        }
    }, [setDashboardReady]);

    useEffect(() => {
        setDashboardReady(false);
        loadDashboard(false);
        const timer = window.setInterval(() => {
            loadDashboard(true);
        }, 30000);
        return () => window.clearInterval(timer);
    }, [loadDashboard, setDashboardReady]);

    const primaryGpu = gpus[0] ?? gpu;
    const gpuName = primaryGpu?.name && primaryGpu.name !== 'N/A'
        ? primaryGpu.name
        : capabilities?.gpu_name || 'No GPU detected';
    const gpuUtilization = Number(primaryGpu?.utilization || 0);
    const gpuMemoryUsed = Number(primaryGpu?.memoryUsed || 0);
    const gpuMemoryTotal = Number(primaryGpu?.memoryTotal || 0);
    const gpuMemoryPct = gpuMemoryTotal > 0 ? (gpuMemoryUsed / gpuMemoryTotal) * 100 : 0;

    const totals = useMemo(() => {
        return projects.reduce((acc, project) => {
            acc.projects += 1;
            acc.images += Number(project.images_total || 0);
            acc.labeled += Number(project.labeled_images || 0);
            acc.labels += Number(project.labels_total || 0);
            acc.runs += Number(project.runs_total || 0);
            if ((project.analysis_type || 'classification') === 'detection') {
                acc.detection += 1;
            } else {
                acc.classification += 1;
            }
            return acc;
        }, {
            projects: 0,
            images: 0,
            labeled: 0,
            labels: 0,
            runs: 0,
            detection: 0,
            classification: 0
        });
    }, [projects]);

    const labeledPct = totals.images > 0 ? (totals.labeled / totals.images) * 100 : 0;

    const activeRuns = useMemo(
        () => trainingRuns
            .filter((run) => run.status === 'starting' || run.status === 'training')
            .sort((a, b) => (b.startedAt || 0) - (a.startedAt || 0)),
        [trainingRuns]
    );

    const latestRuns = useMemo(
        () => trainingRuns
            .slice()
            .sort((a, b) => (b.startedAt || 0) - (a.startedAt || 0))
            .slice(0, 5),
        [trainingRuns]
    );

    const recentProjects = useMemo(
        () => projects.slice().sort((a, b) => (
            new Date(b.updated_at || b.created_at).getTime() - new Date(a.updated_at || a.created_at).getTime()
        )).slice(0, 6),
        [projects]
    );

    const firstProject = projects[0];

    return (
        <div className="animate-fade-in space-y-6">
            <header className="space-y-4">
                <div className="flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
                    <div>
                        <div className="text-xs uppercase tracking-[0.6em] text-haze">Overview</div>
                        <h2 className="text-4xl md:text-5xl font-semibold tracking-tight">Main Dashboard</h2>
                        <p className="text-secondary text-sm mt-2">
                            Workspace status, annotation progress, and active training signals.
                        </p>
                    </div>
                    <div className="flex flex-wrap gap-2">
                        <Button
                            variant="secondary"
                            onClick={() => loadDashboard(true)}
                            disabled={loading || refreshing}
                        >
                            <RefreshCcw size={14} />
                            {refreshing ? 'Refreshing...' : 'Refresh Data'}
                        </Button>
                        <Button variant="secondary" onClick={() => navigate('/projects')}>
                            <FolderOpen size={14} />
                            Open Projects
                        </Button>
                        <Button
                            variant="primary"
                            onClick={() => {
                                if (firstProject?.id) {
                                    navigate(`/projects/${firstProject.id}/train/setup`);
                                } else {
                                    navigate('/projects');
                                }
                            }}
                        >
                            <PlayCircle size={14} />
                            Start Training
                        </Button>
                    </div>
                </div>
                <div className="flex items-center justify-between text-xs text-secondary">
                    <span>{error ? error : 'Dashboard synced with engine endpoints.'}</span>
                    <span>Last update: {toLocaleDate(lastSyncAt)}</span>
                </div>
                <div className="relative h-[2px] bg-graphite-700">
                    <span className="absolute left-0 -top-[3px] h-[6px] w-24 bg-accent-primary" />
                </div>
            </header>

            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                <Card>
                    <div className="flex items-start justify-between gap-3">
                        <div>
                            <div className="text-secondary text-xs uppercase tracking-[0.3em]">Engine</div>
                            <div className="mt-3 flex items-center gap-2">
                                <StatusPill
                                    status={isConnected ? 'running' : 'error'}
                                    label={String(status || 'offline').toUpperCase()}
                                    pulsing={isConnected}
                                />
                            </div>
                        </div>
                        <Activity className="text-accent-primary" size={20} />
                    </div>
                    <div className="mt-3 text-xs text-secondary">
                        Frameworks: {(capabilities?.frameworks || ['ultralytics']).join(', ')}
                    </div>
                    <div className="mt-1 text-xs text-secondary">
                        Engine version: {capabilities?.version || '0.1.0'}
                    </div>
                </Card>

                <Card>
                    <div className="flex items-start justify-between gap-3">
                        <div>
                            <div className="text-secondary text-xs uppercase tracking-[0.3em]">Primary GPU</div>
                            <div className="mt-2 text-xl font-semibold">{gpuUtilization.toFixed(0)}%</div>
                        </div>
                        <Cpu className="text-accent-primary" size={20} />
                    </div>
                    <div className="mt-2 text-sm text-bone/90">{gpuName}</div>
                    <div className="mt-2 text-xs text-secondary">
                        VRAM {formatNumber(gpuMemoryUsed)} / {formatNumber(gpuMemoryTotal)} MB ({formatPercent(gpuMemoryPct)})
                    </div>
                </Card>

                <Card>
                    <div className="flex items-start justify-between gap-3">
                        <div>
                            <div className="text-secondary text-xs uppercase tracking-[0.3em]">Projects</div>
                            <div className="mt-2 text-xl font-semibold">{formatNumber(totals.projects)}</div>
                        </div>
                        <LayoutGrid className="text-accent-primary" size={20} />
                    </div>
                    <div className="mt-2 text-xs text-secondary">
                        Classification: {formatNumber(totals.classification)} | Detection: {formatNumber(totals.detection)}
                    </div>
                    <div className="mt-1 text-xs text-secondary">Total runs: {formatNumber(totals.runs)}</div>
                </Card>

                <Card>
                    <div className="flex items-start justify-between gap-3">
                        <div>
                            <div className="text-secondary text-xs uppercase tracking-[0.3em]">Label Coverage</div>
                            <div className="mt-2 text-xl font-semibold">{formatPercent(labeledPct)}</div>
                        </div>
                        <CheckCircle2 className="text-accent-primary" size={20} />
                    </div>
                    <div className="mt-2 text-xs text-secondary">
                        Labeled images: {formatNumber(totals.labeled)} / {formatNumber(totals.images)}
                    </div>
                    <div className="mt-1 text-xs text-secondary">Total annotations: {formatNumber(totals.labels)}</div>
                </Card>
            </div>

            <div className="grid gap-4 xl:grid-cols-12">
                <Card className="xl:col-span-5 space-y-4">
                    <div className="flex items-center justify-between">
                        <h3 className="text-lg font-semibold">Training Activity</h3>
                        <StatusPill
                            status={activeRuns.length > 0 ? 'running' : 'idle'}
                            label={activeRuns.length > 0 ? `${activeRuns.length} ACTIVE` : 'NO ACTIVE RUNS'}
                            pulsing={activeRuns.length > 0}
                        />
                    </div>
                    {latestRuns.length === 0 ? (
                        <div className="text-secondary text-sm">No runs loaded yet in this session.</div>
                    ) : (
                        <div className="space-y-2">
                            {latestRuns.map((run) => (
                                <button
                                    key={run.runId}
                                    type="button"
                                    className="w-full rounded-xl border border-graphite-700/70 bg-graphite-900/70 px-3 py-2 text-left transition-colors hover:border-accent-primary/60"
                                    onClick={() => {
                                        if (!run.projectId) return;
                                        navigate(`/projects/${run.projectId}/train/live/${run.runId}`);
                                    }}
                                >
                                    <div className="flex items-center justify-between gap-3">
                                        <span className="truncate text-sm font-semibold">{run.name || run.runId}</span>
                                        <StatusPill
                                            status={classifyRunStatus(run.status)}
                                            label={String(run.status || 'idle').toUpperCase()}
                                        />
                                    </div>
                                    <div className="mt-1 text-xs text-secondary">
                                        {run.projectId || 'No project'} · epoch {run.currentEpoch}/{run.totalEpochs}
                                    </div>
                                </button>
                            ))}
                        </div>
                    )}
                </Card>

                <Card className="xl:col-span-7 space-y-4">
                    <div className="flex items-center justify-between">
                        <h3 className="text-lg font-semibold">Recent Projects</h3>
                        <Button variant="ghost" onClick={() => navigate('/projects')}>
                            View All
                        </Button>
                    </div>
                    {loading ? (
                        <div className="text-secondary text-sm">Loading project overview...</div>
                    ) : recentProjects.length === 0 ? (
                        <div className="text-secondary text-sm">No projects available yet.</div>
                    ) : (
                        <div className="space-y-2">
                            {recentProjects.map((project) => {
                                const imagesTotal = Number(project.images_total || 0);
                                const labeledImages = Number(project.labeled_images || 0);
                                const progress = imagesTotal > 0 ? (labeledImages / imagesTotal) * 100 : 0;
                                return (
                                    <button
                                        key={project.id}
                                        type="button"
                                        className="w-full rounded-xl border border-graphite-700/70 bg-graphite-900/70 px-3 py-2 text-left transition-colors hover:border-accent-primary/60"
                                        onClick={() => navigate(`/projects/${project.id}/dataset`)}
                                    >
                                        <div className="flex items-center justify-between gap-3">
                                            <div className="min-w-0">
                                                <div className="truncate text-sm font-semibold">{project.name}</div>
                                                <div className="text-xs text-secondary">
                                                    {project.analysis_type || 'classification'} · classes {project.classes_total || 0} · runs {project.runs_total || 0}
                                                </div>
                                            </div>
                                            <div className="text-right text-xs text-secondary">
                                                <div>{labeledImages}/{imagesTotal || 0} labeled</div>
                                                <div>{formatPercent(progress)}</div>
                                            </div>
                                        </div>
                                        <div className="mt-2 h-1.5 rounded-full bg-graphite-700/80">
                                            <div
                                                className="h-full rounded-full bg-accent-primary"
                                                style={{ width: `${Math.max(0, Math.min(100, progress))}%` }}
                                            />
                                        </div>
                                        <div className="mt-1 text-[11px] text-secondary">
                                            Updated: {toLocaleDate(project.updated_at || project.created_at)} · Last run: {toLocaleDate(project.last_run_at)}
                                        </div>
                                    </button>
                                );
                            })}
                        </div>
                    )}
                </Card>
            </div>

            <Card className="space-y-3">
                <div className="flex items-center gap-2">
                    <Boxes size={16} className="text-accent-primary" />
                    <h3 className="text-sm uppercase tracking-[0.3em] text-secondary">Operational Snapshot</h3>
                </div>
                <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4 text-sm">
                    <div className="rounded-xl border border-graphite-700/70 bg-graphite-900/70 px-3 py-2">
                        <div className="text-secondary text-xs uppercase tracking-[0.25em]">Tracked GPUs</div>
                        <div className="mt-1 font-semibold">{gpus.length > 0 ? gpus.length : 1}</div>
                    </div>
                    <div className="rounded-xl border border-graphite-700/70 bg-graphite-900/70 px-3 py-2">
                        <div className="text-secondary text-xs uppercase tracking-[0.25em]">Session Runs</div>
                        <div className="mt-1 font-semibold">{trainingRuns.length}</div>
                    </div>
                    <div className="rounded-xl border border-graphite-700/70 bg-graphite-900/70 px-3 py-2">
                        <div className="text-secondary text-xs uppercase tracking-[0.25em]">Pending Labels</div>
                        <div className="mt-1 font-semibold">{formatNumber(Math.max(0, totals.images - totals.labeled))}</div>
                    </div>
                    <div className="rounded-xl border border-graphite-700/70 bg-graphite-900/70 px-3 py-2">
                        <div className="text-secondary text-xs uppercase tracking-[0.25em]">Active Training</div>
                        <div className="mt-1 font-semibold">{activeRuns.length}</div>
                    </div>
                </div>
            </Card>
        </div>
    );
};
