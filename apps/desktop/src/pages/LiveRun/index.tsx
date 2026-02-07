import { useParams } from 'react-router-dom';
import { useTrainingStore } from '../../store/trainingStore';
import { TrainingChart } from '../../components/training/TrainingChart';
import { GpuCharts } from '../../components/training/GpuCharts';
import { LogViewer } from '../../components/training/LogViewer';
import { Card } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { StatusPill } from '../../components/ui/StatusPill';

export const LiveRun = () => {
    const { runId, projectId } = useParams();
    const { status, currentEpoch, totalEpochs, metrics } = useTrainingStore();

    // Derived stats
    const lastMetric = metrics[metrics.length - 1];
    const formatMetric = (value?: number) => (typeof value === 'number' ? value.toFixed(4) : '---');
    const loss = formatMetric(lastMetric?.loss);
    const acc = formatMetric(lastMetric?.accuracy);

    return (
        <div className="live-run animate-fade-in space-y-6">
            <header className="flex justify-between items-center mb-6">
                <div>
                    <h2 className="text-2xl font-bold">Training Run <span className="text-accent">{runId?.slice(0, 8)}</span></h2>
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
                            <TrainingChart />
                        </div>
                    </Card>
                </div>
                <div className="col-span-1">
                    <Card className="h-full flex flex-col">
                        <h3 className="mb-4">Logs</h3>
                        <LogViewer />
                    </Card>
                </div>
            </div>

            <Card className="h-full flex flex-col">
                <h3 className="mb-4">GPU Usage</h3>
                <GpuCharts />
            </Card>
        </div>
    );
};
