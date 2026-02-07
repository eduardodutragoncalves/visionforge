import { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { Card } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { Input } from '../../components/ui/Input';
import { Select } from '../../components/ui/Select';
import { useEngineStore } from '../../store/engineStore';
import { ENGINE_HTTP_URL } from '../../lib/engine';

export const TrainSetup = () => {
    const { projectId } = useParams();
    const navigate = useNavigate();
    const gpu = useEngineStore(state => state.gpu);

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

    // Advisors Logic
    useEffect(() => {
        if (target === 'local') {
            const vramRequiredEst = config.batch_size * 0.15; // Rough heuristic (GB) for YoloV8n
            const vramAvailable = (gpu.memoryTotal || 10) / 1024; // Convert MB to GB (Mock is MB)

            if (vramRequiredEst > vramAvailable * 0.8) {
                setAdvice("Warning: High batch size for available VRAM. OOM likely.");
            } else {
                setAdvice(null);
            }
        }
    }, [config.batch_size, gpu.memoryTotal, target]);

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
                                    <div className="font-mono">{gpu.name || 'Detecting...'}</div>
                                </div>
                                <div>
                                    <div className="text-secondary text-sm">VRAM</div>
                                    <div className="font-mono">{gpu.memoryUsed} / {gpu.memoryTotal} MB</div>
                                </div>
                                <div className="h-2 bg-gray-800 rounded overflow-hidden mt-1">
                                    <div
                                        className="h-full bg-accent-primary"
                                        style={{ width: `${(gpu.memoryUsed / (gpu.memoryTotal || 1)) * 100}%` }}
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
