import { useState, useEffect, useMemo } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { Card } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { Input } from '../../components/ui/Input';
import { Select } from '../../components/ui/Select';
import { useEngineStore } from '../../store/engineStore';
import { useTrainingStore } from '../../store/trainingStore';
import { ENGINE_HTTP_URL } from '../../lib/engine';

export const TrainSetup = () => {
    const { projectId } = useParams();
    const navigate = useNavigate();
    const { gpu, gpus } = useEngineStore(state => ({ gpu: state.gpu, gpus: state.gpus }));
    const activeRuns = useTrainingStore(state =>
        Object.values(state.runs).filter((run) =>
            run.projectId === projectId &&
            (run.status === 'starting' || run.status === 'training')
        )
    );

    const [loading, setLoading] = useState(false);
    const [config, setConfig] = useState({
        epochs: 50,
        batch_size: 16,
        base_model: 'yolov8n.pt',
        framework: 'ultralytics',
        img_size: 640
    });
    const [datasetTask, setDatasetTask] = useState<'classification' | 'detection' | null>(null);

    const [target, setTarget] = useState('local');
    const [runtime, setRuntime] = useState<'local' | 'conda' | 'docker'>('local');
    const [runtimeName, setRuntimeName] = useState('');
    const [advice, setAdvice] = useState<string | null>(null);
    const [gpuSelection, setGpuSelection] = useState<'gpu0' | 'gpu1' | 'dual'>('gpu0');
    const [runHistory, setRunHistory] = useState<any[]>([]);
    const [historyLoading, setHistoryLoading] = useState(false);
    const [splitTrainPct, setSplitTrainPct] = useState(80);
    const [splitValPct, setSplitValPct] = useState(20);
    const [splitTestPct, setSplitTestPct] = useState(0);

    const gpuDevices = gpuSelection === 'dual' ? [0, 1] : [gpuSelection === 'gpu1' ? 1 : 0];
    const vramScale = gpuSelection === 'dual' ? 2 : 1;
    const primaryGpu = gpus[0] ?? gpu;
    const displayMemoryTotal = (primaryGpu.memoryTotal || 0) * vramScale;
    const displayMemoryUsed = (primaryGpu.memoryUsed || 0) * vramScale;

    // Advisors Logic
    useEffect(() => {
        if (target === 'local') {
            const vramRequiredEst = config.batch_size * 0.15; // Rough heuristic (GB) for YoloV8n
            const vramAvailable = ((primaryGpu.memoryTotal || 10) * vramScale) / 1024;

            if (vramRequiredEst > vramAvailable * 0.8) {
                setAdvice("Warning: High batch size for available VRAM. OOM likely.");
            } else {
                setAdvice(null);
            }
        }
    }, [config.batch_size, primaryGpu.memoryTotal, target]);

    useEffect(() => {
        if (!projectId) return;
        const loadDatasetTask = async () => {
            try {
                const res = await fetch(`${ENGINE_HTTP_URL}/projects/${projectId}/dataset`);
                if (!res.ok) return;
                const data = await res.json();
                const task = data?.task === 'classification' ? 'classification' : 'detection';
                setDatasetTask(task);
            } catch {
                // ignore
            }
        };
        loadDatasetTask();
    }, [projectId]);

    const splitTotal = splitTrainPct + splitValPct + splitTestPct;
    const splitValid = splitTotal === 100;

    const loadHistory = async () => {
        if (!projectId) return;
        setHistoryLoading(true);
        try {
            const res = await fetch(`${ENGINE_HTTP_URL}/projects/${projectId}/runs`);
            if (!res.ok) {
                setRunHistory([]);
                return;
            }
            const data = await res.json();
            setRunHistory(Array.isArray(data) ? data : []);
        } catch {
            setRunHistory([]);
        } finally {
            setHistoryLoading(false);
        }
    };

    useEffect(() => {
        loadHistory();
    }, [projectId]);

    const applyRunConfig = (run: any) => {
        const cfg = run?.config || {};
        setConfig((prev) => ({
            ...prev,
            epochs: cfg.epochs ?? prev.epochs,
            batch_size: cfg.batch_size ?? cfg.batch ?? prev.batch_size,
            base_model: cfg.base_model ?? prev.base_model,
            framework: cfg.framework ?? prev.framework,
            img_size: cfg.img_size ?? prev.img_size
        }));
        if (cfg.target) {
            setTarget(cfg.target);
        }
        if (cfg.runtime === 'local' || cfg.runtime === 'conda' || cfg.runtime === 'docker') {
            setRuntime(cfg.runtime);
        }
        if (cfg.runtime_config?.conda_env) {
            setRuntimeName(cfg.runtime_config.conda_env);
        } else if (cfg.runtime_config?.docker_image) {
            setRuntimeName(cfg.runtime_config.docker_image);
        } else {
            setRuntimeName('');
        }
        if (Array.isArray(cfg.gpu_devices) && cfg.gpu_devices.length > 1) {
            setGpuSelection('dual');
        } else if (Array.isArray(cfg.gpu_devices) && cfg.gpu_devices[0] === 1) {
            setGpuSelection('gpu1');
        } else if (Array.isArray(cfg.gpu_devices)) {
            setGpuSelection('gpu0');
        }
        if (typeof cfg.split_train === 'number') {
            setSplitTrainPct(cfg.split_train);
        }
        if (typeof cfg.split_val === 'number') {
            setSplitValPct(cfg.split_val);
        }
        if (typeof cfg.split_test === 'number') {
            setSplitTestPct(cfg.split_test);
        }
    };

    const formatMetric = (value: any) => {
        if (typeof value !== 'number') return '—';
        if (Number.isNaN(value)) return '—';
        return value.toFixed(4);
    };

    const buildIndicators = (run: any) => {
        const metrics = run?.final_metrics || {};
        const items = [
            { label: 'Accuracy', value: metrics.accuracy },
            { label: 'Loss', value: metrics.loss },
            { label: 'mAP50', value: metrics.final_map },
            { label: 'Precision', value: metrics.precision },
            { label: 'Recall', value: metrics.recall }
        ];
        return items.filter((item) => item.value !== undefined && item.value !== null);
    };

    const modelOptions = useMemo(() => {
        if (datasetTask === 'classification') {
            return [
                { value: 'yolo26n-cls.pt', label: 'YOLO26 Nano (cls)' },
                { value: 'yolo26s-cls.pt', label: 'YOLO26 Small (cls)' },
                { value: 'yolo26m-cls.pt', label: 'YOLO26 Medium (cls)' },
                { value: 'yolo26l-cls.pt', label: 'YOLO26 Large (cls)' },
                { value: 'yolo26x-cls.pt', label: 'YOLO26 XLarge (cls)' },
                { value: 'yolo11n-cls.pt', label: 'YOLO11 Nano (cls)' },
                { value: 'yolo11s-cls.pt', label: 'YOLO11 Small (cls)' },
                { value: 'yolo11m-cls.pt', label: 'YOLO11 Medium (cls)' },
                { value: 'yolo11l-cls.pt', label: 'YOLO11 Large (cls)' },
                { value: 'yolo11x-cls.pt', label: 'YOLO11 XLarge (cls)' },
                { value: 'yolov8n-cls.pt', label: 'YOLOv8 Nano (cls)' },
                { value: 'yolov8s-cls.pt', label: 'YOLOv8 Small (cls)' },
                { value: 'yolov8m-cls.pt', label: 'YOLOv8 Medium (cls)' },
                { value: 'yolov8l-cls.pt', label: 'YOLOv8 Large (cls)' },
                { value: 'yolov8x-cls.pt', label: 'YOLOv8 XLarge (cls)' }
            ];
        }
        return [
            { value: 'yolov8n.pt', label: 'YOLOv8 Nano' },
            { value: 'yolov8s.pt', label: 'YOLOv8 Small' },
            { value: 'yolov8m.pt', label: 'YOLOv8 Medium' }
        ];
    }, [datasetTask]);

    useEffect(() => {
        if (modelOptions.length === 0) return;
        const values = new Set(modelOptions.map((option) => option.value));
        if (!values.has(config.base_model)) {
            setConfig((prev) => ({ ...prev, base_model: modelOptions[0].value }));
        }
    }, [modelOptions, config.base_model]);

    const handleStart = async () => {
        setLoading(true);
        try {
            const res = await fetch(`${ENGINE_HTTP_URL}/projects/${projectId}/runs`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    config: {
                        ...config,
                        target,
                        runtime,
                        gpu_devices: gpuDevices,
                        gpu_mode: gpuSelection === 'dual' ? 'multi' : 'single',
                        split_train: datasetTask === 'classification' ? splitTrainPct : undefined,
                        split_val: datasetTask === 'classification' ? splitValPct : undefined,
                        split_test: datasetTask === 'classification' ? splitTestPct : undefined,
                        runtime_config: {
                            conda_env: runtime === 'conda' ? runtimeName : undefined,
                            docker_image: runtime === 'docker' ? runtimeName : undefined
                        }
                    }
                })
            });

            if (res.ok) {
                const data = await res.json();
                navigate(`/projects/${projectId}/train/live/${data.run_id}`);
                loadHistory();
            } else {
                console.error("Failed to start run");
            }
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="train-setup animate-fade-in w-full max-w-none space-y-6 pb-20">
            <header className="mb-8">
                <h2 className="text-3xl font-bold">Training Configuration</h2>
                <p className="text-secondary">Setup experiment hyperparameters for {projectId}</p>
            </header>

            {activeRuns.length > 0 && (
                <Card className="border border-accent-primary/30 bg-graphite-900/60">
                    <div className="flex flex-col gap-4">
                        <div>
                            <div className="text-xs uppercase tracking-[0.4em] text-haze">Trainings in progress</div>
                            <div className="text-secondary text-sm mt-2">
                                Selecione um treino em andamento para acompanhar.
                            </div>
                        </div>
                        <div className="flex flex-wrap gap-2">
                            {activeRuns.map((run) => (
                                <Button
                                    key={run.runId}
                                    variant="secondary"
                                    onClick={() => navigate(`/projects/${projectId}/train/live/${run.runId}`)}
                                >
                                    Voltar para {run.runId.slice(0, 8)}
                                </Button>
                            ))}
                        </div>
                    </div>
                </Card>
            )}

            <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">

                {/* Left Col: Config */}
                <div className="space-y-6 xl:col-span-8">
                    <Card>
                        <h3 className="mb-4 text-xl">Model Parameters</h3>
                        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
                            <Select
                                label="Framework"
                                options={[
                                    { value: 'ultralytics', label: 'YOLOv8 (Ultralytics)' },
                                    { value: 'tensorflow', label: 'TensorFlow (Mobilenet)' }
                                ]}
                                value={config.framework}
                                onChange={e => setConfig({ ...config, framework: e.target.value })}
                            />
                            <Select
                                label="Base Model"
                                options={modelOptions}
                                value={config.base_model}
                                onChange={e => setConfig({ ...config, base_model: e.target.value })}
                            />
                            <Input
                                label="Epochs"
                                type="number"
                                value={config.epochs}
                                onChange={e => setConfig({ ...config, epochs: parseInt(e.target.value) })}
                            />
                            <Input
                                label="Image Size"
                                type="number"
                                value={config.img_size}
                                onChange={e => setConfig({ ...config, img_size: parseInt(e.target.value) })}
                            />
                            {datasetTask === 'classification' && (
                                <>
                                    <Input
                                        label="Split Train (%)"
                                        type="number"
                                        min={0}
                                        max={100}
                                        value={splitTrainPct}
                                        onChange={e => setSplitTrainPct(Math.max(0, Math.min(100, Number(e.target.value) || 0)))}
                                    />
                                    <Input
                                        label="Split Val (%)"
                                        type="number"
                                        min={0}
                                        max={100}
                                        value={splitValPct}
                                        onChange={e => setSplitValPct(Math.max(0, Math.min(100, Number(e.target.value) || 0)))}
                                    />
                                    <Input
                                        label="Split Test (%)"
                                        type="number"
                                        min={0}
                                        max={100}
                                        value={splitTestPct}
                                        onChange={e => setSplitTestPct(Math.max(0, Math.min(100, Number(e.target.value) || 0)))}
                                    />
                                    <div className="md:col-span-2 xl:col-span-3 text-xs text-secondary">
                                        Total: {splitTotal}% {splitValid ? '' : '· Ajuste para 100%.'}
                                    </div>
                                </>
                            )}
                            {datasetTask && datasetTask !== 'classification' && (
                                <div className="md:col-span-2 xl:col-span-3 text-xs text-secondary">
                                    Split de treino/validação segue a estrutura do dataset para {datasetTask}.
                                </div>
                            )}
                        </div>
                    </Card>

                    <Card>
                        <h3 className="mb-4 text-xl">Compute Settings</h3>
                        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
                            <Input
                                label="Batch Size"
                                type="number"
                                value={config.batch_size}
                                onChange={e => setConfig({ ...config, batch_size: parseInt(e.target.value) })}
                            />
                            <Select
                                label="Execution Target"
                                options={[
                                    { value: 'local', label: 'Local Machine' },
                                    { value: 'runpod', label: 'RunPod Cloud' },
                                    { value: 'colab', label: 'Google Colab Export' },
                                ]}
                                value={target}
                                onChange={e => setTarget(e.target.value)}
                            />

                            {target === 'local' && (
                                <>
                                    <div className="md:col-span-2 xl:col-span-3 pt-4 border-t border-white/10 mt-2">
                                        <h4 className="text-sm font-bold mb-2">Local Runtime Environment</h4>
                                    </div>
                                    <Select
                                        label="GPU Selection"
                                        options={[
                                            { value: 'gpu0', label: 'GPU 0 (RTX 3090)' },
                                            { value: 'gpu1', label: 'GPU 1 (RTX 3090)' },
                                            { value: 'dual', label: 'Both GPUs (Data Parallel)' }
                                        ]}
                                        value={gpuSelection}
                                        onChange={e => setGpuSelection(e.target.value as typeof gpuSelection)}
                                    />
                                    <Select
                                        label="Environment Type"
                                        options={[
                                            { value: 'local', label: 'System Python (venv)' },
                                            { value: 'conda', label: 'Anaconda Environment' },
                                            { value: 'docker', label: 'Docker Container' },
                                        ]}
                                        value={runtime}
                                        onChange={e => setRuntime(e.target.value as any)}
                                    />
                                    {runtime === 'conda' && (
                                        <Input
                                            label="Conda Env Name"
                                            placeholder="e.g. visionforge"
                                            value={runtimeName}
                                            onChange={e => setRuntimeName(e.target.value)}
                                        />
                                    )}
                                    {runtime === 'docker' && (
                                        <Input
                                            label="Docker Image"
                                            placeholder="e.g. ultralytics/ultralytics:latest"
                                            value={runtimeName}
                                            onChange={e => setRuntimeName(e.target.value)}
                                        />
                                    )}
                                </>
                            )}
                        </div>

                        {/* Advisor Banner */}
                        {advice && target === 'local' && (
                            <div className="mt-4 p-3 bg-warning/10 border border-warning/30 rounded text-warning text-sm flex items-center">
                                <span className="mr-2">⚠️</span> {advice}
                            </div>
                        )}

                        {target === 'local' && !advice && (
                            <div className="mt-4 p-3 bg-success/10 border border-success/30 rounded text-success text-sm flex items-center">
                                <span className="mr-2">✓</span> Local GPU is adequate for this configuration.
                            </div>
                        )}
                    </Card>
                </div>

                {/* Right Col: Summary & Actions */}
                <div className="space-y-6 xl:col-span-4">
                    <Card className="sticky top-6">
                        <h3 className="mb-4 text-lg font-bold">Hardware Status</h3>
                        {target === 'local' ? (
                            <div className="space-y-4">
                                <div>
                                    <div className="text-secondary text-sm">GPU</div>
                                    <div className="font-mono">{gpuSelection === 'dual' ? 'GPU 0 + GPU 1' : `GPU ${gpuSelection === 'gpu1' ? 1 : 0}`}</div>
                                </div>
                                <div>
                                    <div className="text-secondary text-sm">VRAM</div>
                                    <div className="font-mono">{displayMemoryUsed} / {displayMemoryTotal} MB</div>
                                </div>
                                <div className="h-2 bg-gray-800 rounded overflow-hidden mt-1">
                                    <div
                                        className="h-full bg-accent-primary"
                                        style={{ width: `${(displayMemoryUsed / (displayMemoryTotal || 1)) * 100}%` }}
                                    ></div>
                                </div>
                            </div>
                        ) : (
                            <div className="text-secondary italic">
                                Using remote compute. Local hardware will not be utilized.
                            </div>
                        )}

                        <div className="mt-8 pt-6 border-t border-white/10">
                            <Button
                                variant="primary"
                                className="w-full h-12 text-lg"
                                onClick={handleStart}
                                disabled={loading || (datasetTask === 'classification' && !splitValid)}
                            >
                                {loading ? 'Starting...' : 'Start Training'}
                            </Button>
                        </div>
                    </Card>
                </div>
            </div>

            <Card>
                <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                    <div>
                        <h3 className="text-xl font-semibold">Training History</h3>
                        <p className="text-secondary text-sm">Runs previously executed for this project.</p>
                    </div>
                    <Button variant="ghost" onClick={loadHistory} disabled={historyLoading}>
                        {historyLoading ? 'Refreshing...' : 'Refresh'}
                    </Button>
                </div>
                {historyLoading ? (
                    <div className="mt-4 text-secondary text-sm">Loading history...</div>
                ) : runHistory.length === 0 ? (
                    <div className="mt-4 text-secondary text-sm">No runs yet.</div>
                ) : (
                    <div className="mt-4 grid gap-4 md:grid-cols-2 2xl:grid-cols-3">
                        {runHistory.map((run) => {
                            const indicators = buildIndicators(run);
                            return (
                                <div key={run.id} className="border border-graphite-700/70 bg-graphite-900/60 rounded-lg p-4 space-y-3 h-full flex flex-col">
                                    <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                                        <div>
                                            <div className="text-xs uppercase tracking-[0.4em] text-haze">Run</div>
                                            <div className="text-bone font-mono">{run.name || run.id}</div>
                                            <div className="text-xs text-secondary mt-1">
                                                Status: {run.status || 'unknown'}
                                            </div>
                                        </div>
                                        <div className="flex flex-wrap gap-2">
                                            <Button size="sm" variant="secondary" onClick={() => applyRunConfig(run)}>
                                                Duplicar configuração
                                            </Button>
                                            <Button
                                                size="sm"
                                                variant="ghost"
                                                onClick={() => navigate(`/projects/${projectId}/train/live/${run.id}`)}
                                            >
                                                Abrir
                                            </Button>
                                        </div>
                                    </div>
                                    <div className="flex flex-wrap gap-2 text-xs uppercase tracking-[0.3em] text-secondary">
                                        {run.config?.base_model && (
                                            <span className="stat-chip">Model: <strong>{run.config.base_model}</strong></span>
                                        )}
                                        {run.config?.epochs && (
                                            <span className="stat-chip">Epochs: <strong>{run.config.epochs}</strong></span>
                                        )}
                                        {(run.config?.batch_size || run.config?.batch) && (
                                            <span className="stat-chip">
                                                Batch: <strong>{run.config?.batch_size ?? run.config?.batch}</strong>
                                            </span>
                                        )}
                                        {run.config?.img_size && (
                                            <span className="stat-chip">Image Size: <strong>{run.config.img_size}</strong></span>
                                        )}
                                        {(run.config?.split_train !== undefined ||
                                            run.config?.split_val !== undefined ||
                                            run.config?.split_test !== undefined) && (
                                            <span className="stat-chip">
                                                Split:{' '}
                                                <strong>
                                                    {run.config?.split_train ?? 0}% / {run.config?.split_val ?? 0}% / {run.config?.split_test ?? 0}%
                                                </strong>
                                            </span>
                                        )}
                                        {run.created_at && (
                                            <span className="stat-chip">Started: <strong>{new Date(run.created_at).toLocaleString()}</strong></span>
                                        )}
                                    </div>
                                    <div className="flex flex-wrap gap-2 text-xs uppercase tracking-[0.3em] text-secondary">
                                        {indicators.length === 0 ? (
                                            <span className="text-secondary">No metrics recorded.</span>
                                        ) : (
                                            indicators.map((item) => (
                                                <span key={`${run.id}-${item.label}`} className="stat-chip">
                                                    {item.label}: <strong>{formatMetric(item.value)}</strong>
                                                </span>
                                            ))
                                        )}
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                )}
            </Card>
        </div>
    );
};
