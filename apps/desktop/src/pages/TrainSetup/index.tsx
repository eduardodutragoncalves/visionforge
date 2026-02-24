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
    SlidersHorizontal,
    Trash2
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
    id?: string;
    run_id?: string;
    onnx_exported?: boolean;
    exports?: {
        onnx?: boolean;
        onnx_path?: string | null;
    };
    validation?: {
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
    };
}

interface ValidationSummary {
    total_images: number;
    tested_images: number;
    untested_images: number;
    coverage_pct: number;
    correct: number;
    incorrect: number;
    pending: number;
    labeled: number;
    accuracy_pct: number | null;
    last_tested_at: string | null;
}

interface DatasetVersionSummary {
    id: string;
    name: string;
    task?: string;
    source_count?: number;
    selected_count?: number;
    created_at?: string;
    rules?: {
        sample_size?: number | null;
        balance_classes?: boolean;
        orientation?: string;
        resolution?: string;
        seed?: number | null;
    };
    class_counts?: Array<{
        class_id: number;
        class_name?: string;
        count: number;
    }>;
}

type DatasetVersionOrientation = 'any' | 'landscape' | 'portrait' | 'square';
type DatasetVersionResolution = 'any' | 'hd' | 'fhd' | 'qhd' | 'uhd';
type DatasetTask = 'classification' | 'detection';

type NumericConfigField = 'epochs' | 'img_size';
type HistoryStatusFilter = 'all' | 'running' | 'completed' | 'failed' | 'other';
type HistoryValidationFilter = 'all' | 'validated' | 'not_validated';
type HistoryOnnxFilter = 'all' | 'onnx' | 'without_onnx';
type HistorySortBy =
    | 'newest'
    | 'oldest'
    | 'train_accuracy_desc'
    | 'validation_accuracy_desc'
    | 'validation_coverage_desc'
    | 'name_asc';

const isInProgressStatus = (status?: string) => {
    const normalized = String(status || '').toLowerCase();
    return normalized === 'queued' || normalized === 'running' || normalized === 'starting' || normalized === 'training';
};

const getRunStatusMeta = (status?: string): { label: string; pill: StatusType; icon: typeof Clock3 } => {
    const normalized = String(status || '').toLowerCase();
    if (normalized === 'completed') return { label: 'Completed', pill: 'success', icon: CheckCircle2 };
    if (normalized === 'failed' || normalized === 'error') return { label: 'Failed', pill: 'error', icon: AlertTriangle };
    if (isInProgressStatus(normalized)) return { label: 'In progress', pill: 'running', icon: Clock3 };
    return { label: normalized || 'Unknown', pill: 'idle', icon: Activity };
};

const normalizeValidation = (validation?: VariantSummary['validation']): ValidationSummary => {
    const total_images = Math.max(0, Number(validation?.total_images || 0));
    const tested_images = Math.max(0, Number(validation?.tested_images || 0));
    const correct = Math.max(0, Number(validation?.correct || 0));
    const incorrect = Math.max(0, Number(validation?.incorrect || 0));
    const pending = Math.max(0, Number(validation?.pending || 0));
    const labeled = Math.max(0, Number(validation?.labeled || correct + incorrect));
    const inferredCoverage = total_images > 0 ? (tested_images / total_images) * 100 : 0;
    const coverage_pct = typeof validation?.coverage_pct === 'number' ? validation.coverage_pct : inferredCoverage;
    return {
        total_images,
        tested_images,
        untested_images: Math.max(0, Number(validation?.untested_images ?? (total_images - tested_images))),
        coverage_pct: Math.max(0, Math.min(100, coverage_pct)),
        correct,
        incorrect,
        pending,
        labeled,
        accuracy_pct: typeof validation?.accuracy_pct === 'number' ? validation.accuracy_pct : null,
        last_tested_at: validation?.last_tested_at ?? null
    };
};

const formatPct = (value?: number | null, digits = 1) => {
    if (typeof value !== 'number' || Number.isNaN(value)) return '—';
    return `${value.toFixed(digits)}%`;
};

const toAccuracyPercent = (value?: number | null) => {
    if (typeof value !== 'number' || Number.isNaN(value)) return null;
    if (value <= 1) return value * 100;
    return value;
};

const resolveDatasetFormatByFramework = (framework: string) => {
    if (framework === 'ultralytics') return 'yolo';
    return 'yolo';
};

const parseIntegerInput = (value: string) => {
    const trimmed = value.trim();
    if (!trimmed) return null;
    const parsed = Number(trimmed);
    if (!Number.isFinite(parsed)) return null;
    return Math.trunc(parsed);
};

const clampPercentage = (value: number) => Math.max(0, Math.min(100, value));

const BATCH_SIZE_MIN = 16;
const BATCH_SIZE_MAX = 2048;
const BATCH_SIZE_STEP = 16;

const normalizeBatchSize = (value: unknown) => {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return BATCH_SIZE_MIN;
    const clamped = Math.max(BATCH_SIZE_MIN, Math.min(BATCH_SIZE_MAX, Math.round(parsed)));
    return Math.round(clamped / BATCH_SIZE_STEP) * BATCH_SIZE_STEP;
};

const asMetricNumber = (value: unknown): number | null => {
    if (typeof value !== 'number' || Number.isNaN(value)) return null;
    return value;
};

const resolveRunTask = (run: RunHistoryItem, fallbackTask: DatasetTask | null): DatasetTask => {
    const modelName = String(run?.config?.base_model || run?.config?.model || '').toLowerCase();
    if (modelName.includes('-cls')) return 'classification';
    if (fallbackTask) return fallbackTask;

    const metrics = run?.final_metrics || {};
    const hasMapMetric = asMetricNumber(metrics.final_map) !== null || asMetricNumber(metrics.map50) !== null;
    const hasAccuracyMetric = asMetricNumber(metrics.accuracy) !== null;
    if (hasMapMetric) return 'detection';
    if (hasAccuracyMetric) return 'classification';
    return 'detection';
};

const getRunPrimaryMetric = (run: RunHistoryItem, task: DatasetTask) => {
    const metrics = run?.final_metrics || {};
    if (task === 'classification') {
        return {
            key: 'accuracy',
            label: 'Train accuracy',
            value: asMetricNumber(metrics.accuracy),
            format: 'percent' as const
        };
    }
    return {
        key: 'final_map',
        label: 'Train mAP50',
        value: asMetricNumber(metrics.final_map) ?? asMetricNumber(metrics.map50),
        format: 'decimal' as const
    };
};

export const TrainSetup = () => {
    const { projectId } = useParams();
    const navigate = useNavigate();
    const resetRun = useTrainingStore(state => state.resetRun);
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
        batch_size: BATCH_SIZE_MIN,
        base_model: 'yolov8n.pt',
        framework: 'ultralytics',
        img_size: 640
    });
    const [datasetTask, setDatasetTask] = useState<DatasetTask | null>(null);

    const [target, setTarget] = useState('local');
    const [runtime, setRuntime] = useState<'local' | 'conda' | 'docker'>('local');
    const [runtimeName, setRuntimeName] = useState('');
    const [advice, setAdvice] = useState<string | null>(null);
    const [gpuSelection, setGpuSelection] = useState<'gpu0' | 'gpu1' | 'dual'>('gpu0');
    const [runHistory, setRunHistory] = useState<RunHistoryItem[]>([]);
    const [runOnnxMap, setRunOnnxMap] = useState<Record<string, boolean>>({});
    const [runValidationMap, setRunValidationMap] = useState<Record<string, ValidationSummary>>({});
    const [historyLoading, setHistoryLoading] = useState(false);
    const [deletingRunId, setDeletingRunId] = useState<string | null>(null);
    const [historyQuery, setHistoryQuery] = useState('');
    const [historyStatusFilter, setHistoryStatusFilter] = useState<HistoryStatusFilter>('all');
    const [historyValidationFilter, setHistoryValidationFilter] = useState<HistoryValidationFilter>('all');
    const [historyOnnxFilter, setHistoryOnnxFilter] = useState<HistoryOnnxFilter>('all');
    const [historySortBy, setHistorySortBy] = useState<HistorySortBy>('newest');
    const [splitTrainPct, setSplitTrainPct] = useState(80);
    const [splitValPct, setSplitValPct] = useState(20);
    const [splitTestPct, setSplitTestPct] = useState(0);
    const [datasetVersions, setDatasetVersions] = useState<DatasetVersionSummary[]>([]);
    const [datasetVersionsLoading, setDatasetVersionsLoading] = useState(false);
    const [datasetVersionError, setDatasetVersionError] = useState<string | null>(null);
    const [datasetVersionNotice, setDatasetVersionNotice] = useState<string | null>(null);
    const [selectedDatasetVersionId, setSelectedDatasetVersionId] = useState('');
    const [versionName, setVersionName] = useState('');
    const [versionSampleSize, setVersionSampleSize] = useState('');
    const [versionBalanceClasses, setVersionBalanceClasses] = useState(true);
    const [versionOrientation, setVersionOrientation] = useState<DatasetVersionOrientation>('any');
    const [versionResolution, setVersionResolution] = useState<DatasetVersionResolution>('any');
    const [versionSeed, setVersionSeed] = useState('');
    const [creatingDatasetVersion, setCreatingDatasetVersion] = useState(false);
    const datasetFormatForTraining = useMemo(
        () => resolveDatasetFormatByFramework(config.framework),
        [config.framework]
    );

    const handleConfigNumberChange = useCallback((field: NumericConfigField, rawValue: string, min = 1) => {
        const parsed = parseIntegerInput(rawValue);
        if (parsed === null) return;
        setConfig((prev) => ({ ...prev, [field]: Math.max(min, parsed) }));
    }, []);

    const handleSplitChange = useCallback((split: 'train' | 'val' | 'test', rawValue: string) => {
        const parsed = parseIntegerInput(rawValue);
        if (parsed === null) return;
        const next = clampPercentage(parsed);
        if (split === 'train') {
            setSplitTrainPct(next);
            return;
        }
        if (split === 'val') {
            setSplitValPct(next);
            return;
        }
        setSplitTestPct(next);
    }, []);

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

    const loadDatasetVersions = useCallback(async () => {
        if (!projectId) return;
        setDatasetVersionsLoading(true);
        setDatasetVersionError(null);
        try {
            const res = await fetch(`${ENGINE_HTTP_URL}/projects/${projectId}/dataset/versions`);
            if (!res.ok) {
                const message = await res.text();
                throw new Error(message || 'Failed to load dataset versions');
            }
            const payload = await res.json();
            const versions: DatasetVersionSummary[] = Array.isArray(payload?.versions) ? payload.versions : [];
            setDatasetVersions(versions);
            setSelectedDatasetVersionId((current) => {
                if (current && versions.some((version) => version.id === current)) return current;
                return '';
            });
        } catch (error: any) {
            setDatasetVersionError(error?.message || 'Failed to load dataset versions');
            setDatasetVersions([]);
            setSelectedDatasetVersionId('');
        } finally {
            setDatasetVersionsLoading(false);
        }
    }, [projectId]);

    useEffect(() => {
        if (!projectId) return;
        if (datasetTask !== 'classification') {
            setDatasetVersions([]);
            setSelectedDatasetVersionId('');
            setDatasetVersionError(null);
            setDatasetVersionNotice(null);
            return;
        }
        void loadDatasetVersions();
    }, [datasetTask, loadDatasetVersions, projectId]);

    const selectedDatasetVersion = useMemo(
        () => datasetVersions.find((version) => version.id === selectedDatasetVersionId) || null,
        [datasetVersions, selectedDatasetVersionId]
    );

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
            const validationByRun: Record<string, ValidationSummary> = {};

            variantsData.forEach((variant) => {
                const runId = variant?.run_id;
                const hasOnnx = Boolean(variant?.onnx_exported || variant?.exports?.onnx);
                if (runId && hasOnnx) {
                    onnxByRun[runId] = true;
                }
                if (runId) {
                    const candidate = normalizeValidation(variant.validation);
                    const existing = validationByRun[runId];
                    if (!existing) {
                        validationByRun[runId] = candidate;
                    } else {
                        const existingScore = existing.labeled * 1000 + existing.tested_images;
                        const candidateScore = candidate.labeled * 1000 + candidate.tested_images;
                        if (candidateScore > existingScore) {
                            validationByRun[runId] = candidate;
                        }
                    }
                }
            });

            setRunOnnxMap(onnxByRun);
            setRunValidationMap(validationByRun);
            setRunHistory(Array.isArray(runsData) ? runsData : []);
        } catch {
            setRunOnnxMap({});
            setRunValidationMap({});
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
    const validatedRuns = runHistory.filter((run) => (runValidationMap[run.id]?.tested_images || 0) > 0).length;
    const filteredRunHistory = useMemo(() => {
        const query = historyQuery.trim().toLowerCase();
        const parseTimestamp = (value?: string) => {
            const ts = Date.parse(String(value || ''));
            return Number.isNaN(ts) ? 0 : ts;
        };

        const filtered = runHistory.filter((run) => {
            const validation = runValidationMap[run.id];
            const hasValidation = Boolean(validation && validation.tested_images > 0);
            const hasOnnx = Boolean(runOnnxMap[run.id]);
            const normalizedStatus = String(run.status || '').toLowerCase();

            if (historyStatusFilter === 'running' && !isInProgressStatus(normalizedStatus)) return false;
            if (historyStatusFilter === 'completed' && normalizedStatus !== 'completed') return false;
            if (historyStatusFilter === 'failed' && normalizedStatus !== 'failed' && normalizedStatus !== 'error') return false;
            if (
                historyStatusFilter === 'other' &&
                (isInProgressStatus(normalizedStatus) || normalizedStatus === 'completed' || normalizedStatus === 'failed' || normalizedStatus === 'error')
            ) {
                return false;
            }

            if (historyValidationFilter === 'validated' && !hasValidation) return false;
            if (historyValidationFilter === 'not_validated' && hasValidation) return false;

            if (historyOnnxFilter === 'onnx' && !hasOnnx) return false;
            if (historyOnnxFilter === 'without_onnx' && hasOnnx) return false;

            if (query) {
                const searchable = [
                    run.id,
                    run.name,
                    String(run.config?.base_model || ''),
                    String(run.status || '')
                ].join(' ').toLowerCase();
                if (!searchable.includes(query)) return false;
            }
            return true;
        });

        filtered.sort((a, b) => {
            if (historySortBy === 'newest') return parseTimestamp(b.created_at) - parseTimestamp(a.created_at);
            if (historySortBy === 'oldest') return parseTimestamp(a.created_at) - parseTimestamp(b.created_at);
            if (historySortBy === 'name_asc') return String(a.name || a.id).localeCompare(String(b.name || b.id));
            if (historySortBy === 'train_accuracy_desc') {
                const taskA = resolveRunTask(a, datasetTask);
                const taskB = resolveRunTask(b, datasetTask);
                const primaryA = getRunPrimaryMetric(a, taskA).value ?? -Infinity;
                const primaryB = getRunPrimaryMetric(b, taskB).value ?? -Infinity;
                return primaryB - primaryA;
            }
            if (historySortBy === 'validation_accuracy_desc') {
                const aAcc = runValidationMap[a.id]?.accuracy_pct ?? -Infinity;
                const bAcc = runValidationMap[b.id]?.accuracy_pct ?? -Infinity;
                return bAcc - aAcc;
            }
            const aCoverage = runValidationMap[a.id]?.coverage_pct ?? -Infinity;
            const bCoverage = runValidationMap[b.id]?.coverage_pct ?? -Infinity;
            return bCoverage - aCoverage;
        });

        return filtered;
    }, [datasetTask, historyOnnxFilter, historyQuery, historySortBy, historyStatusFilter, historyValidationFilter, runHistory, runOnnxMap, runValidationMap]);

    const applyRunConfig = (run: RunHistoryItem) => {
        const cfg = run?.config || {};
        setConfig((prev) => ({
            ...prev,
            epochs: cfg.epochs ?? prev.epochs,
            batch_size: normalizeBatchSize(cfg.batch_size ?? cfg.batch ?? prev.batch_size),
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
        setSelectedDatasetVersionId(typeof cfg.dataset_version_id === 'string' ? cfg.dataset_version_id : '');
    };

    const formatMetric = (value: any) => {
        if (typeof value !== 'number') return '—';
        if (Number.isNaN(value)) return '—';
        return value.toFixed(4);
    };

    const formatAccuracyBadge = (value?: number | null) => {
        const percent = toAccuracyPercent(value);
        if (percent === null) return '—';
        return `${percent.toFixed(1)}%`;
    };

    const buildIndicators = (run: RunHistoryItem, task: DatasetTask) => {
        const metrics = run?.final_metrics || {};
        const items = task === 'classification'
            ? [
                { key: 'accuracy', label: 'Accuracy', value: metrics.accuracy },
                { key: 'accuracy_top5', label: 'Accuracy top-5', value: metrics.accuracy_top5 },
                { key: 'loss', label: 'Loss', value: metrics.loss }
            ]
            : [
                { key: 'final_map', label: 'mAP50', value: metrics.final_map ?? metrics.map50 },
                { key: 'map5095', label: 'mAP50-95', value: metrics.map5095 },
                { key: 'precision', label: 'Precision', value: metrics.precision },
                { key: 'recall', label: 'Recall', value: metrics.recall },
                { key: 'loss', label: 'Loss', value: metrics.loss }
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
            { value: 'yolo26n.pt', label: 'YOLO26 Nano' },
            { value: 'yolo26s.pt', label: 'YOLO26 Small' },
            { value: 'yolo26m.pt', label: 'YOLO26 Medium' },
            { value: 'yolo26l.pt', label: 'YOLO26 Large' },
            { value: 'yolo26x.pt', label: 'YOLO26 XLarge' },
            { value: 'yolo11n.pt', label: 'YOLO11 Nano' },
            { value: 'yolo11s.pt', label: 'YOLO11 Small' },
            { value: 'yolo11m.pt', label: 'YOLO11 Medium' },
            { value: 'yolo11l.pt', label: 'YOLO11 Large' },
            { value: 'yolo11x.pt', label: 'YOLO11 XLarge' },
            { value: 'yolov8n.pt', label: 'YOLOv8 Nano' },
            { value: 'yolov8s.pt', label: 'YOLOv8 Small' },
            { value: 'yolov8m.pt', label: 'YOLOv8 Medium' },
            { value: 'yolov8l.pt', label: 'YOLOv8 Large' },
            { value: 'yolov8x.pt', label: 'YOLOv8 XLarge' },
            { value: 'yolov5nu.pt', label: 'YOLOv5u Nano' },
            { value: 'yolov5su.pt', label: 'YOLOv5u Small' },
            { value: 'yolov5mu.pt', label: 'YOLOv5u Medium' },
            { value: 'yolov5lu.pt', label: 'YOLOv5u Large' },
            { value: 'yolov5xu.pt', label: 'YOLOv5u XLarge' }
        ];
    }, [datasetTask]);

    useEffect(() => {
        if (modelOptions.length === 0) return;
        const values = new Set(modelOptions.map((option) => option.value));
        if (!values.has(config.base_model)) {
            setConfig((prev) => ({ ...prev, base_model: modelOptions[0].value }));
        }
    }, [modelOptions, config.base_model]);

    const handleCreateDatasetVersion = async () => {
        if (!projectId || datasetTask !== 'classification') return;
        setCreatingDatasetVersion(true);
        setDatasetVersionError(null);
        setDatasetVersionNotice(null);
        const parsedSampleSize = parseIntegerInput(versionSampleSize);
        const parsedSeed = parseIntegerInput(versionSeed);
        try {
            const res = await fetch(`${ENGINE_HTTP_URL}/projects/${projectId}/dataset/versions/classification`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: versionName.trim() || null,
                    sample_size: parsedSampleSize ?? null,
                    balance_classes: versionBalanceClasses,
                    orientation: versionOrientation,
                    resolution: versionResolution,
                    seed: parsedSeed ?? null
                })
            });
            if (!res.ok) {
                const message = await res.text();
                throw new Error(message || 'Failed to create dataset version');
            }
            const payload = await res.json();
            await loadDatasetVersions();
            if (payload?.id) {
                setSelectedDatasetVersionId(payload.id);
            }
            setDatasetVersionNotice(
                `Dataset version created with ${payload?.selected_count ?? 0} image(s).`
            );
        } catch (error: any) {
            setDatasetVersionError(error?.message || 'Failed to create dataset version');
        } finally {
            setCreatingDatasetVersion(false);
        }
    };

    const handleDeleteDatasetVersion = async (versionId: string) => {
        if (!projectId || !versionId) return;
        const target = datasetVersions.find((item) => item.id === versionId);
        const confirmed = window.confirm(
            `Delete dataset version ${target?.name || versionId}?\n\nThis will not remove original images.`
        );
        if (!confirmed) return;
        setDatasetVersionError(null);
        setDatasetVersionNotice(null);
        try {
            const res = await fetch(`${ENGINE_HTTP_URL}/projects/${projectId}/dataset/versions/${versionId}`, {
                method: 'DELETE'
            });
            if (!res.ok) {
                const message = await res.text();
                throw new Error(message || 'Failed to delete dataset version');
            }
            await loadDatasetVersions();
            if (selectedDatasetVersionId === versionId) {
                setSelectedDatasetVersionId('');
            }
            setDatasetVersionNotice('Dataset version deleted.');
        } catch (error: any) {
            setDatasetVersionError(error?.message || 'Failed to delete dataset version');
        }
    };

    const handleStart = async () => {
        if (!projectId) return;
        setLoading(true);
        const runConfig = {
            ...config,
            batch_size: normalizeBatchSize(config.batch_size),
            task: datasetTask || undefined,
            target,
            runtime,
            dataset_format: datasetFormatForTraining,
            dataset_version_id: selectedDatasetVersionId || undefined,
            gpu_devices: gpuDevices,
            gpu_mode: gpuSelection === 'dual' ? 'multi' : 'single',
            split_train: splitTrainPct,
            split_val: splitValPct,
            split_test: splitTestPct,
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

    const handleDeleteRun = async (run: RunHistoryItem) => {
        if (!projectId || !run?.id) return;
        const confirmed = window.confirm(
            `Delete training run ${run.name || run.id}?\n\nThis removes the run and all generated resources (including related variants/exports).`
        );
        if (!confirmed) return;
        setDeletingRunId(run.id);
        try {
            const res = await fetch(`${ENGINE_HTTP_URL}/projects/${projectId}/runs/${run.id}`, {
                method: 'DELETE'
            });
            if (!res.ok) {
                const message = await res.text();
                window.alert(`Could not delete the training run.\n${message}`);
                return;
            }
            resetRun(run.id);
            setRunHistory((prev) => prev.filter((item) => item.id !== run.id));
            setRunOnnxMap((prev) => {
                const next = { ...prev };
                delete next[run.id];
                return next;
            });
            setRunValidationMap((prev) => {
                const next = { ...prev };
                delete next[run.id];
                return next;
            });
            await loadHistory();
        } catch (error) {
            console.error(error);
            window.alert('Error while deleting training run.');
        } finally {
            setDeletingRunId(null);
        }
    };

    const trainSortLabel = datasetTask === 'classification' ? 'Train accuracy (desc)' : 'Train mAP50 (desc)';
    const validationSortLabel = datasetTask === 'classification' ? 'Validation accuracy (desc)' : 'Validation score (desc)';
    const validationCardLabel = datasetTask === 'classification' ? 'Validation accuracy' : 'Validation score';
    const historyDescription = datasetTask === 'classification'
        ? 'Compact cards with training and validation accuracy highlights.'
        : 'Compact cards with detection metrics and validation score highlights.';

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

            <div className="grid grid-cols-2 gap-4 xl:grid-cols-5">
                <Card className="border border-graphite-700/70">
                    <div className="flex items-center justify-between text-secondary text-xs uppercase tracking-[0.3em]">
                        <span>History</span>
                        <History size={14} />
                    </div>
                    <div className="mt-2 text-2xl font-mono">{totalRuns}</div>
                </Card>
                <Card className="border border-accent-primary/25 bg-accent-primary/5">
                    <div className="flex items-center justify-between text-secondary text-xs uppercase tracking-[0.3em]">
                        <span>In progress</span>
                        <Clock3 size={14} />
                    </div>
                    <div className="mt-2 text-2xl font-mono text-accent-primary">{runningRuns}</div>
                </Card>
                <Card className="border border-accent-success/25 bg-accent-success/5">
                    <div className="flex items-center justify-between text-secondary text-xs uppercase tracking-[0.3em]">
                        <span>Completed</span>
                        <CheckCircle2 size={14} />
                    </div>
                    <div className="mt-2 text-2xl font-mono text-accent-success">{completedRuns}</div>
                </Card>
                <Card className="border border-accent-warning/25 bg-accent-warning/5">
                    <div className="flex items-center justify-between text-secondary text-xs uppercase tracking-[0.3em]">
                        <span>ONNX ready</span>
                        <Download size={14} />
                    </div>
                    <div className="mt-2 text-2xl font-mono text-accent-warning">{onnxRuns}</div>
                </Card>
                <Card className="border border-accent-success/25 bg-accent-success/5">
                    <div className="flex items-center justify-between text-secondary text-xs uppercase tracking-[0.3em]">
                        <span>With validation</span>
                        <CheckCircle2 size={14} />
                    </div>
                    <div className="mt-2 text-2xl font-mono text-accent-success">{validatedRuns}</div>
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
                                    Select a running training run to monitor.
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
                                    Open {run.name}
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
                                <p className="text-xs text-secondary uppercase tracking-[0.24em]">Architecture, dimensions and split</p>
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
                                min={1}
                                value={config.epochs}
                                onChange={e => handleConfigNumberChange('epochs', e.target.value, 1)}
                            />
                            <Input
                                label="Image Size"
                                type="number"
                                min={1}
                                value={config.img_size}
                                onChange={e => handleConfigNumberChange('img_size', e.target.value, 1)}
                            />
                            {datasetTask && (
                                <>
                                    <Input
                                        label="Split Train (%)"
                                        type="number"
                                        min={0}
                                        max={100}
                                        value={splitTrainPct}
                                        onChange={e => handleSplitChange('train', e.target.value)}
                                    />
                                    <Input
                                        label="Split Val (%)"
                                        type="number"
                                        min={0}
                                        max={100}
                                        value={splitValPct}
                                        onChange={e => handleSplitChange('val', e.target.value)}
                                    />
                                    <Input
                                        label="Split Test (%)"
                                        type="number"
                                        min={0}
                                        max={100}
                                        value={splitTestPct}
                                        onChange={e => handleSplitChange('test', e.target.value)}
                                    />
                                    <div className="md:col-span-2 xl:col-span-3 rounded-lg border border-graphite-700/70 bg-graphite-900 px-3 py-2 text-xs text-secondary">
                                        Total: {splitTotal}% {splitValid ? '· Configuration valid.' : '· Adjust to 100%.'}
                                    </div>
                                    {datasetTask === 'detection' && (
                                        <div className="md:col-span-2 xl:col-span-3 rounded-lg border border-graphite-700/70 bg-graphite-900 px-3 py-2 text-xs text-secondary">
                                            Detection split is applied during training. If images already contain explicit split tags, those tags take priority.
                                        </div>
                                    )}
                                </>
                            )}
                            <div className="md:col-span-2 xl:col-span-3 rounded-lg border border-graphite-700/70 bg-graphite-900 px-3 py-2 text-xs text-secondary">
                                Annotation format resolved during training by framework: <span className="text-bone font-semibold uppercase">{datasetFormatForTraining}</span>.
                            </div>
                            {datasetTask === 'classification' && (
                                <div className="md:col-span-2 xl:col-span-3 rounded-lg border border-graphite-700/70 bg-graphite-900 px-3 py-2 text-xs text-secondary">
                                    YOLOv5u models are available for detection in this Ultralytics release. Classification currently uses YOLOv8, YOLO11 and YOLO26 families.
                                </div>
                            )}
                            {datasetTask === 'classification' && (
                                <div className="md:col-span-2 xl:col-span-3 rounded-lg border border-accent-primary/25 bg-accent-primary/5 px-3 py-3">
                                    <div className="flex flex-wrap items-center justify-between gap-2">
                                        <div className="text-xs uppercase tracking-[0.24em] text-secondary">Dataset versions</div>
                                        <div className="flex items-center gap-2">
                                            <Button size="sm" variant="ghost" onClick={loadDatasetVersions} disabled={datasetVersionsLoading}>
                                                <RefreshCcw size={12} className="mr-1" />
                                                {datasetVersionsLoading ? 'Refreshing...' : 'Refresh'}
                                            </Button>
                                            {selectedDatasetVersionId && (
                                                <Button
                                                    size="sm"
                                                    variant="ghost"
                                                    className="text-error hover:text-error"
                                                    onClick={() => handleDeleteDatasetVersion(selectedDatasetVersionId)}
                                                >
                                                    <Trash2 size={12} className="mr-1" />
                                                    Delete selected
                                                </Button>
                                            )}
                                        </div>
                                    </div>

                                    <div className="mt-3">
                                        <Select
                                            label="Training dataset"
                                            value={selectedDatasetVersionId}
                                            onChange={(e) => setSelectedDatasetVersionId(e.target.value)}
                                            options={[
                                                { value: '', label: 'Default labeled dataset (all images)' },
                                                ...datasetVersions.map((version) => ({
                                                    value: version.id,
                                                    label: `${version.name} (${version.selected_count ?? 0})`
                                                }))
                                            ]}
                                        />
                                    </div>

                                    <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-2">
                                        <Input
                                            label="Version name (optional)"
                                            placeholder="Balanced FHD sample"
                                            value={versionName}
                                            onChange={(e) => setVersionName(e.target.value)}
                                        />
                                        <Input
                                            label="Sample size (optional)"
                                            type="number"
                                            min={1}
                                            value={versionSampleSize}
                                            onChange={(e) => setVersionSampleSize(e.target.value)}
                                        />
                                        <Select
                                            label="Orientation"
                                            value={versionOrientation}
                                            onChange={(e) => setVersionOrientation(e.target.value as DatasetVersionOrientation)}
                                            options={[
                                                { value: 'any', label: 'Any orientation' },
                                                { value: 'landscape', label: 'Landscape only' },
                                                { value: 'portrait', label: 'Portrait only' },
                                                { value: 'square', label: 'Square only' }
                                            ]}
                                        />
                                        <Select
                                            label="Resolution tier"
                                            value={versionResolution}
                                            onChange={(e) => setVersionResolution(e.target.value as DatasetVersionResolution)}
                                            options={[
                                                { value: 'any', label: 'Any resolution' },
                                                { value: 'hd', label: 'HD (1280x720)' },
                                                { value: 'fhd', label: 'Full HD (1920x1080)' },
                                                { value: 'qhd', label: '2K/QHD (2560x1440)' },
                                                { value: 'uhd', label: '4K/UHD (3840x2160)' }
                                            ]}
                                        />
                                        <Input
                                            label="Random seed (optional)"
                                            type="number"
                                            min={0}
                                            value={versionSeed}
                                            onChange={(e) => setVersionSeed(e.target.value)}
                                        />
                                        <label className="flex items-center gap-2 rounded-lg border border-graphite-700/70 bg-graphite-900/70 px-3 py-2 text-sm text-secondary">
                                            <input
                                                type="checkbox"
                                                checked={versionBalanceClasses}
                                                onChange={(e) => setVersionBalanceClasses(e.target.checked)}
                                            />
                                            Balance classes
                                        </label>
                                    </div>

                                    <div className="mt-3 flex flex-wrap gap-2">
                                        <Button
                                            size="sm"
                                            variant="secondary"
                                            onClick={handleCreateDatasetVersion}
                                            disabled={creatingDatasetVersion}
                                        >
                                            {creatingDatasetVersion ? 'Creating...' : 'Create dataset version'}
                                        </Button>
                                    </div>

                                    {datasetVersionError && (
                                        <div className="mt-3 rounded border border-accent-error/40 bg-accent-error/10 px-2 py-2 text-xs text-accent-error">
                                            {datasetVersionError}
                                        </div>
                                    )}
                                    {datasetVersionNotice && (
                                        <div className="mt-3 rounded border border-accent-success/35 bg-accent-success/10 px-2 py-2 text-xs text-accent-success">
                                            {datasetVersionNotice}
                                        </div>
                                    )}
                                    {selectedDatasetVersion && (
                                        <div className="mt-3 rounded border border-graphite-700/70 bg-graphite-900/70 px-2 py-2 text-xs text-secondary">
                                            Selected: <span className="text-bone font-semibold">{selectedDatasetVersion.name}</span> ·
                                            {' '}Images <span className="text-bone font-semibold">{selectedDatasetVersion.selected_count ?? 0}</span>
                                            {' '}of <span className="text-bone font-semibold">{selectedDatasetVersion.source_count ?? 0}</span>.
                                        </div>
                                    )}
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
                                <p className="text-xs text-secondary uppercase tracking-[0.24em]">Local or remote execution</p>
                            </div>
                        </div>
                        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
                            <div className="md:col-span-2 xl:col-span-3 rounded-lg border border-graphite-700/70 bg-graphite-900 px-3 py-3">
                                <div className="mb-2 flex items-center justify-between">
                                    <label className="text-sm text-secondary">Batch Size</label>
                                    <span className="rounded-full border border-accent-primary/45 bg-accent-primary/10 px-2 py-0.5 text-xs font-semibold text-bone">
                                        {config.batch_size}
                                    </span>
                                </div>
                                <input
                                    type="range"
                                    min={BATCH_SIZE_MIN}
                                    max={BATCH_SIZE_MAX}
                                    step={BATCH_SIZE_STEP}
                                    value={normalizeBatchSize(config.batch_size)}
                                    onChange={(e) => {
                                        const nextValue = normalizeBatchSize(e.target.value);
                                        setConfig((prev) => ({ ...prev, batch_size: nextValue }));
                                    }}
                                    className="w-full accent-accent-primary"
                                />
                                <div className="mt-1 flex items-center justify-between text-[10px] uppercase tracking-[0.2em] text-secondary">
                                    <span>{BATCH_SIZE_MIN}</span>
                                    <span>Step {BATCH_SIZE_STEP}</span>
                                    <span>{BATCH_SIZE_MAX}</span>
                                </div>
                            </div>
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
                                <div className="text-xs text-secondary uppercase tracking-[0.24em]">{memoryPct.toFixed(1)}% used</div>
                            </div>
                        ) : (
                            <div className="text-secondary italic">
                                Using remote compute. Local hardware will not be utilized.
                            </div>
                        )}

                        <div className="mt-6 rounded-lg border border-graphite-700/70 bg-graphite-900 px-3 py-3 text-xs text-secondary">
                            <div className="uppercase tracking-[0.24em] mb-1">Quick summary</div>
                            <div>Model: <span className="text-bone">{config.base_model}</span></div>
                            <div>Epochs: <span className="text-bone">{config.epochs}</span></div>
                            <div>Batch: <span className="text-bone">{config.batch_size}</span></div>
                            {datasetTask === 'classification' && (
                                <div>
                                    Dataset version:{' '}
                                    <span className="text-bone">
                                        {selectedDatasetVersion?.name || 'Default labeled dataset'}
                                    </span>
                                </div>
                            )}
                        </div>

                        <div className="mt-8 pt-6 border-t border-white/10">
                            <Button
                                variant="primary"
                                className="w-full h-12 text-lg"
                                onClick={handleStart}
                                disabled={loading || !splitValid}
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
                            <p className="text-secondary text-sm">Compact cards with training and validation accuracy highlights.</p>
                        </div>
                    </div>
                    <Button variant="ghost" onClick={loadHistory} disabled={historyLoading}>
                        <RefreshCcw size={14} className="mr-2" />
                        {historyLoading ? 'Refreshing...' : 'Refresh'}
                    </Button>
                </div>

                {!historyLoading && runHistory.length > 0 && (
                    <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-5">
                        <Input
                            label="Search"
                            placeholder="Run name, ID, model"
                            value={historyQuery}
                            onChange={(e) => setHistoryQuery(e.target.value)}
                        />
                        <Select
                            label="Status"
                            value={historyStatusFilter}
                            onChange={(e) => setHistoryStatusFilter(e.target.value as HistoryStatusFilter)}
                            options={[
                                { value: 'all', label: 'All statuses' },
                                { value: 'running', label: 'In progress' },
                                { value: 'completed', label: 'Completed' },
                                { value: 'failed', label: 'Failed' },
                                { value: 'other', label: 'Other' }
                            ]}
                        />
                        <Select
                            label="Validation"
                            value={historyValidationFilter}
                            onChange={(e) => setHistoryValidationFilter(e.target.value as HistoryValidationFilter)}
                            options={[
                                { value: 'all', label: 'All' },
                                { value: 'validated', label: 'Validated only' },
                                { value: 'not_validated', label: 'Not validated' }
                            ]}
                        />
                        <Select
                            label="ONNX"
                            value={historyOnnxFilter}
                            onChange={(e) => setHistoryOnnxFilter(e.target.value as HistoryOnnxFilter)}
                            options={[
                                { value: 'all', label: 'All runs' },
                                { value: 'onnx', label: 'ONNX exported' },
                                { value: 'without_onnx', label: 'Without ONNX export' }
                            ]}
                        />
                        <Select
                            label="Sort"
                            value={historySortBy}
                            onChange={(e) => setHistorySortBy(e.target.value as HistorySortBy)}
                            options={[
                                { value: 'newest', label: 'Newest first' },
                                { value: 'oldest', label: 'Oldest first' },
                                { value: 'train_accuracy_desc', label: 'Train accuracy (desc)' },
                                { value: 'validation_accuracy_desc', label: 'Validation accuracy (desc)' },
                                { value: 'validation_coverage_desc', label: 'Validation coverage (desc)' },
                                { value: 'name_asc', label: 'Name (A-Z)' }
                            ]}
                        />
                    </div>
                )}

                {historyLoading ? (
                    <div className="mt-4 text-secondary text-sm flex items-center gap-2">
                        <Activity size={14} className="animate-pulse" />
                        Loading history...
                    </div>
                ) : runHistory.length === 0 ? (
                    <div className="mt-4 text-secondary text-sm">No runs yet.</div>
                ) : filteredRunHistory.length === 0 ? (
                    <div className="mt-4 rounded-lg border border-graphite-700/70 bg-graphite-900/50 px-3 py-3 text-sm text-secondary">
                        No runs match the current filters.
                    </div>
                ) : (
                    <>
                        <div className="mt-4 text-xs text-secondary uppercase tracking-[0.24em]">
                            Showing {filteredRunHistory.length} of {runHistory.length} runs
                        </div>
                        <div className="mt-3 grid gap-3 md:grid-cols-2 2xl:grid-cols-3">
                            {filteredRunHistory.map((run) => {
                                const indicators = buildIndicators(run);
                                const extraIndicators = indicators.filter((item) => item.label !== 'Accuracy');
                                const statusMeta = getRunStatusMeta(run.status);
                                const StatusIcon = statusMeta.icon;
                                const onnxExported = Boolean(runOnnxMap[run.id]);
                                const runValidation = runValidationMap[run.id] || null;
                                const isInProgress = isInProgressStatus(run.status);
                                const isDeleting = deletingRunId === run.id;
                                return (
                                    <div
                                        key={run.id}
                                        className={`rounded-lg p-3 space-y-2.5 h-full flex flex-col border ${
                                            onnxExported
                                                ? 'border-accent-warning/45 bg-accent-warning/5'
                                                : 'border-graphite-700/70 bg-graphite-900/60'
                                        }`}
                                    >
                                        <div className="flex items-start justify-between gap-2">
                                            <div className="min-w-0">
                                                <div className="text-[10px] uppercase tracking-[0.35em] text-haze">Run</div>
                                                <div className="text-sm text-bone font-mono truncate">{run.name || run.id}</div>
                                                <div className="text-[11px] text-secondary mt-1 flex items-center gap-1.5">
                                                    <StatusIcon size={11} />
                                                    <span>{run.status || 'unknown'}</span>
                                                </div>
                                            </div>
                                            <div className="flex flex-col items-end gap-1.5">
                                                <StatusPill status={statusMeta.pill} label={statusMeta.label.toUpperCase()} pulsing={statusMeta.pill === 'running'} />
                                                {onnxExported && (
                                                    <div className="inline-flex items-center gap-1 rounded border border-accent-warning/50 bg-accent-warning/10 px-1.5 py-0.5 text-[10px] uppercase tracking-[0.18em] text-accent-warning">
                                                        <Download size={10} />
                                                        ONNX Exported
                                                    </div>
                                                )}
                                            </div>
                                        </div>

                                        <div className="grid grid-cols-2 gap-2">
                                            <div className="rounded-lg border border-accent-primary/30 bg-accent-primary/10 px-2 py-2">
                                                <div className="text-[10px] uppercase tracking-[0.2em] text-secondary">Train accuracy</div>
                                                <div className="mt-1 text-lg font-mono text-accent-primary">{formatAccuracyBadge(run.final_metrics?.accuracy)}</div>
                                            </div>
                                            <div className={`rounded-lg border px-2 py-2 ${
                                                runValidation && runValidation.tested_images > 0
                                                    ? 'border-accent-success/35 bg-accent-success/10'
                                                    : 'border-graphite-700/70 bg-graphite-900/70'
                                            }`}>
                                                <div className="text-[10px] uppercase tracking-[0.2em] text-secondary">Validation accuracy</div>
                                                <div className="mt-1 text-lg font-mono text-accent-success">{formatPct(runValidation?.accuracy_pct)}</div>
                                            </div>
                                        </div>

                                        <div className="flex flex-wrap gap-1.5 text-[10px] uppercase tracking-[0.16em] text-secondary">
                                            {run.config?.base_model && (
                                                <span className="rounded border border-graphite-700/70 bg-graphite-900 px-1.5 py-1">
                                                    Model <strong className="text-bone">{run.config.base_model}</strong>
                                                </span>
                                            )}
                                            {run.config?.epochs && (
                                                <span className="rounded border border-graphite-700/70 bg-graphite-900 px-1.5 py-1">
                                                    Epochs <strong className="text-bone">{run.config.epochs}</strong>
                                                </span>
                                            )}
                                            {(run.config?.batch_size || run.config?.batch) && (
                                                <span className="rounded border border-graphite-700/70 bg-graphite-900 px-1.5 py-1">
                                                    Batch <strong className="text-bone">{run.config?.batch_size ?? run.config?.batch}</strong>
                                                </span>
                                            )}
                                            {run.config?.img_size && (
                                                <span className="rounded border border-graphite-700/70 bg-graphite-900 px-1.5 py-1">
                                                    Img <strong className="text-bone">{run.config.img_size}</strong>
                                                </span>
                                            )}
                                            {(run.config?.split_train !== undefined ||
                                                run.config?.split_val !== undefined ||
                                                run.config?.split_test !== undefined) && (
                                                    <span className="rounded border border-graphite-700/70 bg-graphite-900 px-1.5 py-1">
                                                        Split <strong className="text-bone">{run.config?.split_train ?? 0}%/{run.config?.split_val ?? 0}%/{run.config?.split_test ?? 0}%</strong>
                                                    </span>
                                                )}
                                            {run.config?.dataset_version_id && (
                                                <span className="rounded border border-accent-primary/35 bg-accent-primary/10 px-1.5 py-1">
                                                    Dataset <strong className="text-bone">{run.config.dataset_version_id}</strong>
                                                </span>
                                            )}
                                            {run.created_at && (
                                                <span className="rounded border border-graphite-700/70 bg-graphite-900 px-1.5 py-1">
                                                    Started <strong className="text-bone">{new Date(run.created_at).toLocaleString()}</strong>
                                                </span>
                                            )}
                                        </div>

                                        <div className="flex flex-wrap gap-1.5 text-[10px] uppercase tracking-[0.16em] text-secondary">
                                            {extraIndicators.length === 0 ? (
                                                <span className="text-secondary">No additional metrics.</span>
                                            ) : (
                                                extraIndicators.map((item) => (
                                                    <span key={`${run.id}-${item.label}`} className="rounded border border-graphite-700/70 bg-graphite-900 px-1.5 py-1">
                                                        {item.label} <strong className="text-bone">{formatMetric(item.value)}</strong>
                                                    </span>
                                                ))
                                            )}
                                        </div>

                                        <div className={`rounded border px-2.5 py-2 text-[11px] ${
                                            runValidation && runValidation.tested_images > 0
                                                ? 'border-accent-success/35 bg-accent-success/5'
                                                : 'border-graphite-700/70 bg-graphite-900/70'
                                        }`}>
                                            <div className="uppercase tracking-[0.22em] text-secondary mb-1">Validation details</div>
                                            {!runValidation || runValidation.tested_images === 0 ? (
                                                <div className="text-secondary">This run has no validation data yet.</div>
                                            ) : (
                                                <>
                                                    <div className="text-secondary">
                                                        Coverage <span className="text-bone font-semibold">{runValidation.tested_images}/{runValidation.total_images}</span> ({formatPct(runValidation.coverage_pct)})
                                                    </div>
                                                    <div className="mt-1 flex flex-wrap gap-1.5 uppercase tracking-[0.14em]">
                                                        <span className="rounded border border-accent-success/35 bg-accent-success/10 px-1.5 py-1">
                                                            Correct <strong className="text-bone">{runValidation.correct}</strong>
                                                        </span>
                                                        <span className="rounded border border-accent-error/35 bg-accent-error/10 px-1.5 py-1">
                                                            Wrong <strong className="text-bone">{runValidation.incorrect}</strong>
                                                        </span>
                                                        <span className="rounded border border-graphite-700/70 bg-graphite-900 px-1.5 py-1">
                                                            Pending <strong className="text-bone">{runValidation.pending}</strong>
                                                        </span>
                                                    </div>
                                                    <div className="mt-1 text-secondary">
                                                        Last test {runValidation.last_tested_at ? new Date(runValidation.last_tested_at).toLocaleString() : '—'}
                                                    </div>
                                                </>
                                            )}
                                        </div>

                                        <div className="mt-auto flex flex-wrap gap-2 pt-0.5">
                                            <Button size="sm" variant="secondary" onClick={() => applyRunConfig(run)}>
                                                <CopyPlus size={13} className="mr-1" />
                                                Duplicate
                                            </Button>
                                            <Button
                                                size="sm"
                                                variant="ghost"
                                                onClick={() => navigate(`/projects/${projectId}/train/live/${run.id}`)}
                                            >
                                                <ArrowUpRight size={13} className="mr-1" />
                                                Open
                                            </Button>
                                            <Button
                                                size="sm"
                                                variant="ghost"
                                                className="text-error hover:text-error"
                                                onClick={() => handleDeleteRun(run)}
                                                disabled={isInProgress || isDeleting}
                                                title={isInProgress ? 'A training run in progress cannot be deleted.' : 'Delete training run and artifacts'}
                                            >
                                                <Trash2 size={13} className="mr-1" />
                                                {isDeleting ? 'Deleting...' : 'Delete'}
                                            </Button>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </>
                )}
            </Card>
        </div>
    );
};
