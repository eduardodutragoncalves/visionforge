import { useMemo, useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { useTrainingStore } from '../../store/trainingStore';
import { TrainingChart } from '../../components/training/TrainingChart';
import { GpuCharts } from '../../components/training/GpuCharts';
import { LogViewer } from '../../components/training/LogViewer';
import { Card } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { StatusPill } from '../../components/ui/StatusPill';
import { ENGINE_HTTP_URL } from '../../lib/engine';

const formatDuration = (seconds: number | null | undefined) => {
    if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return '---';
    const total = Math.max(0, Math.floor(seconds));
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const secs = total % 60;
    const hh = String(hours).padStart(2, '0');
    const mm = String(minutes).padStart(2, '0');
    const ss = String(secs).padStart(2, '0');
    return `${hh}:${mm}:${ss}`;
};

export const LiveRun = () => {
    const { runId, projectId } = useParams();
    const navigate = useNavigate();
    const runsForProject = useTrainingStore(state =>
        Object.values(state.runs).filter((run) => run.projectId === projectId)
    );
    const activeRuns = useMemo(
        () => runsForProject.filter((run) => run.status === 'starting' || run.status === 'training'),
        [runsForProject]
    );
    const visibleRuns = activeRuns.length > 0 ? activeRuns : runsForProject;
    const run = useTrainingStore(state => (runId ? state.runs[runId] : undefined));
    const [nowTs, setNowTs] = useState(() => Date.now());
    const [stopBusy, setStopBusy] = useState(false);
    const [stopNotice, setStopNotice] = useState<string | null>(null);

    useEffect(() => {
        if (!projectId) return;
        if (runId) return;
        if (visibleRuns.length === 0) return;
        navigate(`/projects/${projectId}/train/live/${visibleRuns[0].runId}`);
    }, [projectId, runId, visibleRuns, navigate]);

    useEffect(() => {
        const timer = window.setInterval(() => setNowTs(Date.now()), 1000);
        return () => window.clearInterval(timer);
    }, []);

    // Derived stats
    const metrics = run?.metrics ?? [];
    const lastMetric = metrics[metrics.length - 1];
    const formatMetric = (value?: number) => (typeof value === 'number' ? value.toFixed(4) : '---');
    const loss = formatMetric(lastMetric?.loss);
    const lr = formatMetric(lastMetric?.lr);
    const status = run?.status ?? 'idle';
    const currentEpoch = run?.currentEpoch ?? 0;
    const totalEpochs = run?.totalEpochs ?? 0;
    const config = run?.config ?? {};
    const runName = run?.name ?? '';
    const modelName = String(config.base_model || config.model || '').toLowerCase();
    const configuredTask = String(config.task || '').toLowerCase();
    const hasMapMetric = typeof lastMetric?.map50 === 'number' && !Number.isNaN(lastMetric.map50);
    const hasAccuracyMetric = typeof lastMetric?.accuracy === 'number' && !Number.isNaN(lastMetric.accuracy);
    const isClassificationRun =
        configuredTask === 'classification' ||
        configuredTask === 'classify' ||
        modelName.includes('-cls') ||
        (!modelName && !hasMapMetric && hasAccuracyMetric);
    const secondaryMetricLabel = isClassificationRun ? 'Accuracy' : 'mAP50';
    const secondaryMetricKey = isClassificationRun ? 'accuracy' : 'map50';
    const secondaryMetricValue = formatMetric(
        isClassificationRun ? lastMetric?.accuracy : lastMetric?.map50
    );
    const elapsedSec = useMemo(() => {
        if (!run?.startedAt) return null;
        const endTs = run.endedAt ?? nowTs;
        return Math.max(0, Math.floor((endTs - run.startedAt) / 1000));
    }, [run?.startedAt, run?.endedAt, nowTs]);
    const etaSec = useMemo(() => {
        if (!run?.startedAt) return null;
        if (!totalEpochs || currentEpoch <= 0) return null;
        if (currentEpoch >= totalEpochs || status === 'completed') return 0;
        if (status !== 'training' && status !== 'starting') return null;
        const elapsed = Math.max(1, Math.floor((nowTs - run.startedAt) / 1000));
        const avgEpochDuration = elapsed / currentEpoch;
        const remainingEpochs = Math.max(0, totalEpochs - currentEpoch);
        return Math.round(remainingEpochs * avgEpochDuration);
    }, [run?.startedAt, totalEpochs, currentEpoch, status, nowTs]);
    const elapsedLabel = formatDuration(elapsedSec);
    const etaLabel = formatDuration(etaSec);
    const configItems = [
        { label: 'Framework', value: config.framework },
        { label: 'Model', value: config.base_model ?? config.model },
        { label: 'Epochs', value: config.epochs },
        { label: 'Batch', value: config.batch_size ?? config.batch },
        { label: 'Image Size', value: config.img_size ?? config.imgsz },
        { label: 'Target', value: config.target },
        { label: 'Runtime', value: config.runtime },
        {
            label: 'Split',
            value:
                config.split_train !== undefined || config.split_val !== undefined || config.split_test !== undefined
                    ? `${config.split_train ?? 0}% / ${config.split_val ?? 0}% / ${config.split_test ?? 0}%`
                    : undefined
        },
        { label: 'GPU', value: Array.isArray(config.gpu_devices) ? config.gpu_devices.join(', ') : undefined },
        { label: 'Conda Env', value: config.runtime_config?.conda_env },
        { label: 'Docker Image', value: config.runtime_config?.docker_image }
    ].filter((item) => item.value !== undefined && item.value !== null && item.value !== '');

    const requestStopRun = async () => {
        if (!runId) return;
        setStopBusy(true);
        setStopNotice(null);
        try {
            const response = await fetch(`${ENGINE_HTTP_URL}/runs/${runId}/stop`, { method: 'POST' });
            if (!response.ok) {
                throw new Error('Failed to request stop.');
            }
            const payload = await response.json();
            setStopNotice(payload?.detail || 'Stop request sent.');
        } catch (error) {
            setStopNotice(error instanceof Error ? error.message : 'Failed to request stop.');
        } finally {
            setStopBusy(false);
        }
    };

    return (
        <div className="live-run animate-fade-in h-full min-h-0 flex flex-col gap-3">
            {visibleRuns.length > 1 && (
                <div className="flex flex-wrap gap-2 shrink-0">
                    {visibleRuns.map((item) => (
                        <button
                            key={`run-tab-${item.runId}`}
                            type="button"
                            className={`tab-button ${item.runId === runId ? 'tab-button-active' : ''}`}
                            onClick={() => {
                                if (!projectId) return;
                                navigate(`/projects/${projectId}/train/live/${item.runId}`);
                            }}
                        >
                            {item.name ? item.name : item.runId.slice(0, 8)}
                        </button>
                    ))}
                </div>
            )}
            <header className="flex flex-wrap justify-between items-center gap-3">
                <div>
                    <h2 className="text-2xl font-bold">
                        Training Run <span className="text-accent">{runName || runId?.slice(0, 8)}</span>
                    </h2>
                    <p className="text-secondary">Project: {projectId}</p>
                </div>
                <div className="flex gap-4 items-center">
                    <StatusPill
                        status={status === 'training' ? 'running' : 'idle'}
                        label={status.toUpperCase()}
                        pulsing={status === 'training'}
                    />
                    <Button
                        variant="secondary"
                        onClick={requestStopRun}
                        disabled={!runId || stopBusy || status === 'completed' || status === 'failed' || status === 'stopped'}
                    >
                        {stopBusy ? 'Requesting...' : 'Stop Training'}
                    </Button>
                </div>
            </header>
            {stopNotice && (
                <div className="text-xs text-secondary">
                    {stopNotice}
                </div>
            )}
            {configItems.length > 0 && (
                <Card className="shrink-0">
                    <div className="flex gap-2 overflow-x-auto pb-1 text-xs uppercase tracking-[0.3em] text-secondary">
                    {configItems.map((item) => (
                        <div key={`config-${item.label}`} className="border border-graphite-700/70 bg-graphite-900 px-3 py-2 rounded shrink-0">
                            <span className="text-haze">{item.label}:</span>{' '}
                            <span className="text-bone">{String(item.value)}</span>
                        </div>
                    ))}
                    </div>
                </Card>
            )}

            <div className="grid grid-cols-2 gap-4 xl:grid-cols-6 shrink-0">
                <Card>
                    <div className="text-secondary text-sm">Epoch</div>
                    <div className="text-2xl font-mono">{currentEpoch} <span className="text-sm text-secondary">/ {totalEpochs}</span></div>
                </Card>
                <Card>
                    <div className="text-secondary text-sm">Current Loss</div>
                    <div className="text-2xl font-mono text-error">{loss}</div>
                </Card>
                <Card>
                    <div className="text-secondary text-sm">{secondaryMetricLabel}</div>
                    <div className="text-2xl font-mono text-accent">{secondaryMetricValue}</div>
                </Card>
                <Card>
                    <div className="text-secondary text-sm">LR</div>
                    <div className="text-2xl font-mono">{lr}</div>
                </Card>
                <Card>
                    <div className="text-secondary text-sm">Elapsed</div>
                    <div className="text-2xl font-mono">{elapsedLabel}</div>
                </Card>
                <Card>
                    <div className="text-secondary text-sm">ETA</div>
                    <div className="text-2xl font-mono">{etaLabel}</div>
                </Card>
            </div>

            <div
                className="flex-1 min-h-0 grid gap-4"
                style={{ gridTemplateRows: 'minmax(0, 0.9fr) minmax(0, 1.1fr)' }}
            >
                <div className="grid grid-cols-12 gap-4 min-h-0">
                    <div className="col-span-12 xl:col-span-8 min-h-0">
                        <Card className="h-full min-h-0 flex flex-col">
                            <h3 className="mb-3">Loss & {secondaryMetricLabel}</h3>
                            <div className="flex-1 min-h-0">
                                {runId ? (
                                    <TrainingChart
                                        runId={runId}
                                        compact
                                        secondaryKey={secondaryMetricKey}
                                        secondaryLabel={secondaryMetricLabel}
                                    />
                                ) : null}
                            </div>
                        </Card>
                    </div>
                    <div className="col-span-12 xl:col-span-4 min-h-0">
                        <Card className="h-full min-h-0 flex flex-col">
                            <h3 className="mb-3">Logs</h3>
                            <div className="flex-1 min-h-0">
                                {runId ? <LogViewer runId={runId} /> : null}
                            </div>
                        </Card>
                    </div>
                </div>

                <Card className="h-full min-h-0 flex flex-col overflow-hidden">
                    <h3 className="mb-3">GPU Usage</h3>
                    <div className="flex-1 min-h-0">
                        <GpuCharts
                            compact
                            chartHeight={72}
                            visibleGpuIndices={
                                Array.isArray(config.gpu_devices) && config.gpu_devices.length > 0
                                    ? config.gpu_devices.map((value: any) => Number(value)).filter((value: number) => Number.isFinite(value))
                                    : undefined
                            }
                        />
                    </div>
                </Card>
            </div>
        </div>
    );
};
