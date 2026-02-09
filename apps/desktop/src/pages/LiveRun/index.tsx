import { useMemo, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { useTrainingStore } from '../../store/trainingStore';
import { TrainingChart } from '../../components/training/TrainingChart';
import { GpuCharts } from '../../components/training/GpuCharts';
import { LogViewer } from '../../components/training/LogViewer';
import { Card } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { StatusPill } from '../../components/ui/StatusPill';

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

    useEffect(() => {
        if (!projectId) return;
        if (runId) return;
        if (visibleRuns.length === 0) return;
        navigate(`/projects/${projectId}/train/live/${visibleRuns[0].runId}`);
    }, [projectId, runId, visibleRuns, navigate]);

    // Derived stats
    const metrics = run?.metrics ?? [];
    const lastMetric = metrics[metrics.length - 1];
    const formatMetric = (value?: number) => (typeof value === 'number' ? value.toFixed(4) : '---');
    const loss = formatMetric(lastMetric?.loss);
    const acc = formatMetric(lastMetric?.accuracy);
    const status = run?.status ?? 'idle';
    const currentEpoch = run?.currentEpoch ?? 0;
    const totalEpochs = run?.totalEpochs ?? 0;
    const config = run?.config ?? {};
    const runName = run?.name ?? '';
    const configItems = [
        { label: 'Framework', value: config.framework },
        { label: 'Model', value: config.base_model },
        { label: 'Epochs', value: config.epochs },
        { label: 'Batch', value: config.batch_size },
        { label: 'Image Size', value: config.img_size },
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

    return (
        <div className="live-run animate-fade-in space-y-6">
            {visibleRuns.length > 1 && (
                <div className="flex flex-wrap gap-2">
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
            <header className="flex justify-between items-center mb-6">
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
                    <Button variant="secondary" onClick={() => console.log('Mock Stop')}>
                        Stop Training
                    </Button>
                </div>
            </header>
            {configItems.length > 0 && (
                <Card className="flex flex-wrap gap-3 text-xs uppercase tracking-[0.3em] text-secondary">
                    {configItems.map((item) => (
                        <div key={`config-${item.label}`} className="border border-graphite-700/70 bg-graphite-900 px-3 py-2 rounded">
                            <span className="text-haze">{item.label}:</span>{' '}
                            <span className="text-bone">{String(item.value)}</span>
                        </div>
                    ))}
                </Card>
            )}

            <div className="grid grid-cols-4 gap-4 mb-6">
                <Card>
                    <div className="text-secondary text-sm">Epoch</div>
                    <div className="text-2xl font-mono">{currentEpoch} <span className="text-sm text-secondary">/ {totalEpochs}</span></div>
                </Card>
                <Card>
                    <div className="text-secondary text-sm">Current Loss</div>
                    <div className="text-2xl font-mono text-error">{loss}</div>
                </Card>
                <Card>
                    <div className="text-secondary text-sm">Accuracy</div>
                    <div className="text-2xl font-mono text-accent">{acc}</div>
                </Card>
                <Card>
                    <div className="text-secondary text-sm">LR</div>
                    <div className="text-2xl font-mono">0.001</div>
                </Card>
            </div>

            <div className="grid grid-cols-3 gap-6" style={{ height: '400px' }}>
                <div className="col-span-2">
                    <Card className="h-full flex flex-col">
                        <h3 className="mb-4">Loss & Accuracy</h3>
                        <div className="flex-1">
                            {runId ? <TrainingChart runId={runId} /> : null}
                        </div>
                    </Card>
                </div>
                <div className="col-span-1">
                    <Card className="h-full flex flex-col">
                        <h3 className="mb-4">Logs</h3>
                        {runId ? <LogViewer runId={runId} /> : null}
                    </Card>
                </div>
            </div>

            <Card className="h-full flex flex-col">
                <h3 className="mb-4">GPU Usage</h3>
                <GpuCharts
                    visibleGpuIndices={
                        Array.isArray(config.gpu_devices) && config.gpu_devices.length > 0
                            ? config.gpu_devices.map((value: any) => Number(value)).filter((value: number) => Number.isFinite(value))
                            : undefined
                    }
                />
            </Card>
        </div>
    );
};
