import { useState, useEffect } from 'react';
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

    const [target, setTarget] = useState('local');
    const [runtime, setRuntime] = useState<'local' | 'conda' | 'docker'>('local');
    const [runtimeName, setRuntimeName] = useState('');
    const [advice, setAdvice] = useState<string | null>(null);
    const [gpuSelection, setGpuSelection] = useState<'gpu0' | 'gpu1' | 'dual'>('gpu0');

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
        <div className="train-setup animate-fade-in max-w-4xl mx-auto space-y-6 pb-20">
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

            <div className="grid grid-cols-3 gap-6">

                {/* Left Col: Config */}
                <div className="col-span-2 space-y-6">
                    <Card>
                        <h3 className="mb-4 text-xl">Model Parameters</h3>
                        <div className="grid grid-cols-2 gap-4">
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
                                options={[
                                    { value: 'yolov8n.pt', label: 'YOLOv8 Nano' },
                                    { value: 'yolov8s.pt', label: 'YOLOv8 Small' },
                                    { value: 'yolov8m.pt', label: 'YOLOv8 Medium' },
                                ]}
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
                        </div>
                    </Card>

                    <Card>
                        <h3 className="mb-4 text-xl">Compute Settings</h3>
                        <div className="grid grid-cols-2 gap-4">
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
                                    <div className="col-span-2 pt-4 border-t border-white/10 mt-2">
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
                <div className="col-span-1 space-y-6">
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
                                disabled={loading}
                            >
                                {loading ? 'Starting...' : 'Start Training'}
                            </Button>
                        </div>
                    </Card>
                </div>
            </div>
        </div>
    );
};
