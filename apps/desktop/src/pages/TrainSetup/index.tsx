import { useState, useEffect, useMemo, useCallback } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
    Activity,
    AlertTriangle,
    ArrowUpRight,
    CheckCircle2,
    Clock3,
    CopyPlus,
    Cpu,
    Download,
    Gauge,
    History,
    Layers,
    PlayCircle,
    RefreshCcw,
    Rocket,
    SlidersHorizontal
} from 'lucide-react';
import { Card } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { Input } from '../../components/ui/Input';
import { Select } from '../../components/ui/Select';
import { StatusPill, type StatusType } from '../../components/ui/StatusPill';
import { useEngineStore } from '../../store/engineStore';
import { useTrainingStore } from '../../store/trainingStore';
import { ENGINE_HTTP_URL } from '../../lib/engine';

interface RunHistoryItem {
    id: string;
    name?: string;
    status?: string;
    created_at?: string;
    config?: Record<string, any>;
    final_metrics?: Record<string, number>;
}

interface VariantSummary {
    run_id?: string;
    onnx_exported?: boolean;
    exports?: {
        onnx?: boolean;
        onnx_path?: string | null;
    };
}

const isInProgressStatus = (status?: string) => {
    const normalized = String(status || '').toLowerCase();
    return normalized === 'queued' || normalized === 'running' || normalized === 'starting' || normalized === 'training';
};

const getRunStatusMeta = (status?: string): { label: string; pill: StatusType; icon: typeof Clock3 } => {
    const normalized = String(status || '').toLowerCase();
    if (normalized === 'completed') return { label: 'Concluido', pill: 'success', icon: CheckCircle2 };
    if (normalized === 'failed' || normalized === 'error') return { label: 'Falhou', pill: 'error', icon: AlertTriangle };
    if (isInProgressStatus(normalized)) return { label: 'Em progresso', pill: 'running', icon: Clock3 };
    return { label: normalized || 'Desconhecido', pill: 'idle', icon: Activity };
};

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
    const [runHistory, setRunHistory] = useState<RunHistoryItem[]>([]);
    const [runOnnxMap, setRunOnnxMap] = useState<Record<string, boolean>>({});
    const [historyLoading, setHistoryLoading] = useState(false);
    const [splitTrainPct, setSplitTrainPct] = useState(80);
    const [splitValPct, setSplitValPct] = useState(20);
    const [splitTestPct, setSplitTestPct] = useState(0);

    const gpuDevices = gpuSelection === 'dual' ? [0, 1] : [gpuSelection === 'gpu1' ? 1 : 0];
    const vramScale = gpuSelection === 'dual' ? 2 : 1;
    const primaryGpu = gpus[0] ?? gpu;
    const displayMemoryTotal = (primaryGpu.memoryTotal || 0) * vramScale;
    const displayMemoryUsed = (primaryGpu.memoryUsed || 0) * vramScale;
    const memoryPct = Math.min(100, (displayMemoryUsed / (displayMemoryTotal || 1)) * 100);

    useEffect(() => {
        if (target === 'local') {
            const vramRequiredEst = config.batch_size * 0.15;
            const vramAvailable = ((primaryGpu.memoryTotal || 10) * vramScale) / 1024;

            if (vramRequiredEst > vramAvailable * 0.8) {
                setAdvice('Warning: High batch size for available VRAM. OOM likely.');
            } else {
                setAdvice(null);
            }
        }
    }, [config.batch_size, primaryGpu.memoryTotal, target, vramScale]);

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

    const loadHistory = useCallback(async () => {
        if (!projectId) return;
        setHistoryLoading(true);
        try {
            const [runsRes, variantsRes] = await Promise.all([
                fetch(`${ENGINE_HTTP_URL}/projects/${projectId}/runs`),
                fetch(`${ENGINE_HTTP_URL}/projects/${projectId}/variants`)
            ]);

            const runsData = runsRes.ok ? await runsRes.json() : [];
            const variantsData: VariantSummary[] = variantsRes.ok ? await variantsRes.json() : [];
            const onnxByRun: Record<string, boolean> = {};

            variantsData.forEach((variant) => {
                const runId = variant?.run_id;
                const hasOnnx = Boolean(variant?.onnx_exported || variant?.exports?.onnx);
                if (runId && hasOnnx) {
                    onnxByRun[runId] = true;
                }
            });

            setRunOnnxMap(onnxByRun);
            setRunHistory(Array.isArray(runsData) ? runsData : []);
        } catch {
            setRunOnnxMap({});
            setRunHistory([]);
        } finally {
            setHistoryLoading(false);
        }
    }, [projectId]);

    useEffect(() => {
        loadHistory();
    }, [loadHistory]);

    const activeRunItems = useMemo(() => {
        const map = new Map<string, { id: string; name: string }>();
        activeRuns.forEach((run) => {
            map.set(run.runId, {
                id: run.runId,
                name: run.name || run.runId.slice(0, 8)
            });
        });
        runHistory
            .filter((run) => isInProgressStatus(run.status))
            .forEach((run) => {
                map.set(run.id, {
                    id: run.id,
                    name: run.name || run.id.slice(0, 8)
                });
            });
        return Array.from(map.values());
    }, [activeRuns, runHistory]);

    const totalRuns = runHistory.length;
    const runningRuns = runHistory.filter((run) => isInProgressStatus(run.status)).length;
    const completedRuns = runHistory.filter((run) => String(run.status || '').toLowerCase() === 'completed').length;
    const onnxRuns = runHistory.filter((run) => runOnnxMap[run.id]).length;

    const applyRunConfig = (run: RunHistoryItem) => {
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

    const buildIndicators = (run: RunHistoryItem) => {
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
        if (!projectId) return;
        setLoading(true);
        const runConfig = {
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
        };

        try {
            const res = await fetch(`${ENGINE_HTTP_URL}/projects/${projectId}/runs`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ config: runConfig })
            });

            if (res.ok) {
                const data = await res.json();
                setRunHistory((prev) => ([
                    {
                        id: data.run_id,
                        name: data.name,
                        status: 'queued',
                        created_at: new Date().toISOString(),
                        config: runConfig
                    },
                    ...prev.filter((run) => run.id !== data.run_id)
                ]));
                await loadHistory();
                navigate(`/projects/${projectId}/train/live/${data.run_id}`);
            } else {
                console.error('Failed to start run');
            }
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="train-setup animate-fade-in w-full max-w-none space-y-6 pb-20">
            <header className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                <div className="flex items-start gap-3">
                    <div className="rounded-xl border border-accent-primary/30 bg-accent-primary/10 p-2">
                        <SlidersHorizontal size={20} className="text-accent-primary" />
                    </div>
                    <div>
                        <h2 className="text-3xl font-bold">Training Configuration</h2>
                        <p className="text-secondary">Setup experiment hyperparameters for {projectId}</p>
                    </div>
                </div>
                <StatusPill status="running" label="TRAIN PIPELINE READY" pulsing />
            </header>

            <div className="grid grid-cols-2 gap-4 xl:grid-cols-4">
                <Card className="border border-graphite-700/70">
                    <div className="flex items-center justify-between text-secondary text-xs uppercase tracking-[0.3em]">
                        <span>Historico</span>
                        <History size={14} />
                    </div>
                    <div className="mt-2 text-2xl font-mono">{totalRuns}</div>
                </Card>
                <Card className="border border-accent-primary/25 bg-accent-primary/5">
                    <div className="flex items-center justify-between text-secondary text-xs uppercase tracking-[0.3em]">
                        <span>Em progresso</span>
                        <Clock3 size={14} />
                    </div>
                    <div className="mt-2 text-2xl font-mono text-accent-primary">{runningRuns}</div>
                </Card>
                <Card className="border border-accent-success/25 bg-accent-success/5">
                    <div className="flex items-center justify-between text-secondary text-xs uppercase tracking-[0.3em]">
                        <span>Concluidos</span>
                        <CheckCircle2 size={14} />
                    </div>
                    <div className="mt-2 text-2xl font-mono text-accent-success">{completedRuns}</div>
                </Card>
                <Card className="border border-accent-warning/25 bg-accent-warning/5">
                    <div className="flex items-center justify-between text-secondary text-xs uppercase tracking-[0.3em]">
                        <span>ONNX pronto</span>
                        <Download size={14} />
                    </div>
                    <div className="mt-2 text-2xl font-mono text-accent-warning">{onnxRuns}</div>
                </Card>
            </div>

            {activeRunItems.length > 0 && (
                <Card className="border border-accent-primary/30 bg-graphite-900/60">
                    <div className="flex flex-col gap-4">
                        <div className="flex items-start gap-3">
                            <div className="rounded-lg border border-accent-primary/30 bg-accent-primary/10 p-2">
                                <Rocket size={16} className="text-accent-primary" />
                            </div>
                            <div>
                                <div className="text-xs uppercase tracking-[0.4em] text-haze">Trainings in progress</div>
                                <div className="text-secondary text-sm mt-2">
                                    Selecione um treino em andamento para acompanhar.
                                </div>
                            </div>
                        </div>
                        <div className="flex flex-wrap gap-2">
                            {activeRunItems.map((run) => (
                                <Button
                                    key={run.id}
                                    variant="secondary"
                                    onClick={() => navigate(`/projects/${projectId}/train/live/${run.id}`)}
                                >
                                    <PlayCircle size={14} className="mr-2" />
                                    Abrir {run.name}
                                </Button>
                            ))}
                        </div>
                    </div>
                </Card>
            )}

            <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
                <div className="space-y-6 xl:col-span-8">
                    <Card>
                        <div className="mb-4 flex items-center gap-3">
                            <div className="rounded-lg border border-accent-primary/30 bg-accent-primary/10 p-2">
                                <Layers size={16} className="text-accent-primary" />
                            </div>
                            <div>
                                <h3 className="text-xl font-semibold">Model Parameters</h3>
                                <p className="text-xs text-secondary uppercase tracking-[0.24em]">Arquitetura, dimensoes e split</p>
                            </div>
                        </div>
                        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
                            <Select
                                label="Framework"
                                options={[
                                    { value: 'ultralytics', label: 'Ultralytics (YOLO)' }
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
                                    <div className="md:col-span-2 xl:col-span-3 rounded-lg border border-graphite-700/70 bg-graphite-900 px-3 py-2 text-xs text-secondary">
                                        Total: {splitTotal}% {splitValid ? '· Configuracao valida.' : '· Ajuste para 100%.'}
                                    </div>
                                </>
                            )}
                            {datasetTask && datasetTask !== 'classification' && (
                                <div className="md:col-span-2 xl:col-span-3 rounded-lg border border-graphite-700/70 bg-graphite-900 px-3 py-2 text-xs text-secondary">
                                    Split de treino/validacao segue a estrutura do dataset para {datasetTask}.
                                </div>
                            )}
                        </div>
                    </Card>

                    <Card>
                        <div className="mb-4 flex items-center gap-3">
                            <div className="rounded-lg border border-accent-primary/30 bg-accent-primary/10 p-2">
                                <Cpu size={16} className="text-accent-primary" />
                            </div>
                            <div>
                                <h3 className="text-xl font-semibold">Compute Settings</h3>
                                <p className="text-xs text-secondary uppercase tracking-[0.24em]">Execucao local ou remota</p>
                            </div>
                        </div>
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
                                    { value: 'colab', label: 'Google Colab Export' }
                                ]}
                                value={target}
                                onChange={e => setTarget(e.target.value)}
                            />

                            {target === 'local' && (
                                <>
                                    <div className="md:col-span-2 xl:col-span-3 pt-4 border-t border-white/10 mt-2">
                                        <h4 className="text-sm font-bold mb-2 uppercase tracking-[0.24em] text-secondary">Local Runtime Environment</h4>
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
                                            { value: 'docker', label: 'Docker Container' }
                                        ]}
                                        value={runtime}
                                        onChange={e => setRuntime(e.target.value as 'local' | 'conda' | 'docker')}
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

                        {advice && target === 'local' && (
                            <div className="mt-4 p-3 bg-warning/10 border border-warning/30 rounded text-warning text-sm flex items-center gap-2">
                                <AlertTriangle size={14} /> {advice}
                            </div>
                        )}

                        {target === 'local' && !advice && (
                            <div className="mt-4 p-3 bg-success/10 border border-success/30 rounded text-success text-sm flex items-center gap-2">
                                <CheckCircle2 size={14} /> Local GPU is adequate for this configuration.
                            </div>
                        )}
                    </Card>
                </div>

                <div className="space-y-6 xl:col-span-4">
                    <Card className="sticky top-6">
                        <div className="flex items-center gap-3 mb-4">
                            <div className="rounded-lg border border-accent-primary/30 bg-accent-primary/10 p-2">
                                <Gauge size={16} className="text-accent-primary" />
                            </div>
                            <h3 className="text-lg font-bold">Hardware Status</h3>
                        </div>
                        {target === 'local' ? (
                            <div className="space-y-4">
                                <div>
                                    <div className="text-secondary text-xs uppercase tracking-[0.24em]">GPU Selection</div>
                                    <div className="font-mono">{gpuSelection === 'dual' ? 'GPU 0 + GPU 1' : `GPU ${gpuSelection === 'gpu1' ? 1 : 0}`}</div>
                                </div>
                                <div>
                                    <div className="text-secondary text-xs uppercase tracking-[0.24em]">VRAM</div>
                                    <div className="font-mono">{displayMemoryUsed} / {displayMemoryTotal} MB</div>
                                </div>
                                <div className="h-2 bg-gray-800 rounded overflow-hidden mt-1">
                                    <div
                                        className="h-full bg-accent-primary"
                                        style={{ width: `${memoryPct}%` }}
                                    />
                                </div>
                                <div className="text-xs text-secondary uppercase tracking-[0.24em]">{memoryPct.toFixed(1)}% usado</div>
                            </div>
                        ) : (
                            <div className="text-secondary italic">
                                Using remote compute. Local hardware will not be utilized.
                            </div>
                        )}

                        <div className="mt-6 rounded-lg border border-graphite-700/70 bg-graphite-900 px-3 py-3 text-xs text-secondary">
                            <div className="uppercase tracking-[0.24em] mb-1">Resumo rapido</div>
                            <div>Modelo: <span className="text-bone">{config.base_model}</span></div>
                            <div>Epocas: <span className="text-bone">{config.epochs}</span></div>
                            <div>Batch: <span className="text-bone">{config.batch_size}</span></div>
                        </div>

                        <div className="mt-8 pt-6 border-t border-white/10">
                            <Button
                                variant="primary"
                                className="w-full h-12 text-lg"
                                onClick={handleStart}
                                disabled={loading || (datasetTask === 'classification' && !splitValid)}
                            >
                                <Rocket size={16} className="mr-2" />
                                {loading ? 'Starting...' : 'Start Training'}
                            </Button>
                        </div>
                    </Card>
                </div>
            </div>

            <Card>
                <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                    <div className="flex items-center gap-3">
                        <div className="rounded-lg border border-accent-primary/30 bg-accent-primary/10 p-2">
                            <History size={16} className="text-accent-primary" />
                        </div>
                        <div>
                            <h3 className="text-xl font-semibold">Training History</h3>
                            <p className="text-secondary text-sm">Runs executed for this project, with export and status highlights.</p>
                        </div>
                    </div>
                    <Button variant="ghost" onClick={loadHistory} disabled={historyLoading}>
                        <RefreshCcw size={14} className="mr-2" />
                        {historyLoading ? 'Refreshing...' : 'Refresh'}
                    </Button>
                </div>
                {historyLoading ? (
                    <div className="mt-4 text-secondary text-sm flex items-center gap-2">
                        <Activity size={14} className="animate-pulse" />
                        Loading history...
                    </div>
                ) : runHistory.length === 0 ? (
                    <div className="mt-4 text-secondary text-sm">No runs yet.</div>
                ) : (
                    <div className="mt-4 grid gap-4 md:grid-cols-2 2xl:grid-cols-3">
                        {runHistory.map((run) => {
                            const indicators = buildIndicators(run);
                            const statusMeta = getRunStatusMeta(run.status);
                            const StatusIcon = statusMeta.icon;
                            const onnxExported = Boolean(runOnnxMap[run.id]);
                            return (
                                <div
                                    key={run.id}
                                    className={`rounded-lg p-4 space-y-3 h-full flex flex-col border ${
                                        onnxExported
                                            ? 'border-accent-warning/50 bg-accent-warning/5'
                                            : 'border-graphite-700/70 bg-graphite-900/60'
                                    }`}
                                >
                                    <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                                        <div>
                                            <div className="text-xs uppercase tracking-[0.4em] text-haze">Run</div>
                                            <div className="text-bone font-mono break-all">{run.name || run.id}</div>
                                            <div className="text-xs text-secondary mt-1 flex items-center gap-2">
                                                <StatusIcon size={12} />
                                                <span>{run.status || 'unknown'}</span>
                                            </div>
                                        </div>
                                        <div className="flex flex-col items-start gap-2">
                                            <StatusPill status={statusMeta.pill} label={statusMeta.label.toUpperCase()} pulsing={statusMeta.pill === 'running'} />
                                            {onnxExported && (
                                                <div className="inline-flex items-center gap-1 border border-accent-warning/50 bg-accent-warning/10 px-2 py-1 rounded text-[10px] uppercase tracking-[0.2em] text-accent-warning">
                                                    <Download size={11} />
                                                    ONNX Exportado
                                                </div>
                                            )}
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-2 gap-2 text-[11px] uppercase tracking-[0.2em] text-secondary">
                                        {run.config?.base_model && (
                                            <div className="rounded border border-graphite-700/70 bg-graphite-900 px-2 py-2">
                                                Model: <span className="text-bone font-semibold">{run.config.base_model}</span>
                                            </div>
                                        )}
                                        {run.config?.epochs && (
                                            <div className="rounded border border-graphite-700/70 bg-graphite-900 px-2 py-2">
                                                Epochs: <span className="text-bone font-semibold">{run.config.epochs}</span>
                                            </div>
                                        )}
                                        {(run.config?.batch_size || run.config?.batch) && (
                                            <div className="rounded border border-graphite-700/70 bg-graphite-900 px-2 py-2">
                                                Batch: <span className="text-bone font-semibold">{run.config?.batch_size ?? run.config?.batch}</span>
                                            </div>
                                        )}
                                        {run.config?.img_size && (
                                            <div className="rounded border border-graphite-700/70 bg-graphite-900 px-2 py-2">
                                                Img: <span className="text-bone font-semibold">{run.config.img_size}</span>
                                            </div>
                                        )}
                                    </div>

                                    {(run.config?.split_train !== undefined ||
                                        run.config?.split_val !== undefined ||
                                        run.config?.split_test !== undefined) && (
                                        <div className="text-[11px] uppercase tracking-[0.2em] text-secondary rounded border border-graphite-700/70 bg-graphite-900 px-2 py-2">
                                            Split: <span className="text-bone font-semibold">{run.config?.split_train ?? 0}% / {run.config?.split_val ?? 0}% / {run.config?.split_test ?? 0}%</span>
                                        </div>
                                    )}
                                    {run.created_at && (
                                        <div className="text-[11px] uppercase tracking-[0.2em] text-secondary rounded border border-graphite-700/70 bg-graphite-900 px-2 py-2">
                                            Started: <span className="text-bone font-semibold">{new Date(run.created_at).toLocaleString()}</span>
                                        </div>
                                    )}

                                    <div className="flex flex-wrap gap-2 text-[11px] uppercase tracking-[0.2em] text-secondary">
                                        {indicators.length === 0 ? (
                                            <span className="text-secondary">No metrics recorded.</span>
                                        ) : (
                                            indicators.map((item) => (
                                                <span key={`${run.id}-${item.label}`} className="rounded border border-graphite-700/70 bg-graphite-900 px-2 py-1">
                                                    {item.label}: <strong className="text-bone">{formatMetric(item.value)}</strong>
                                                </span>
                                            ))
                                        )}
                                    </div>

                                    <div className="mt-auto flex flex-wrap gap-2 pt-1">
                                        <Button size="sm" variant="secondary" onClick={() => applyRunConfig(run)}>
                                            <CopyPlus size={13} className="mr-1" />
                                            Duplicar
                                        </Button>
                                        <Button
                                            size="sm"
                                            variant="ghost"
                                            onClick={() => navigate(`/projects/${projectId}/train/live/${run.id}`)}
                                        >
                                            <ArrowUpRight size={13} className="mr-1" />
                                            Abrir
                                        </Button>
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
