import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useParams } from 'react-router-dom';
import {
    Camera,
    CheckCircle2,
    Clock3,
    FolderOpen,
    ImagePlus,
    Loader2,
    PlayCircle,
    RefreshCcw,
    ScanSearch,
    Target,
    Upload,
    XCircle
} from 'lucide-react';
import { Card } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { Select } from '../../components/ui/Select';
import { StatusPill } from '../../components/ui/StatusPill';
import { ENGINE_HTTP_URL } from '../../lib/engine';

type Verdict = 'correct' | 'incorrect' | null;

interface Prediction {
    box?: number[];
    confidence: number;
    class_name: string;
    class_id: number;
}

interface InferenceResult {
    task?: 'detect' | 'classify';
    predictions: Prediction[];
    latency_ms: number;
    fps_est: number;
}

interface HistoryVariantResult {
    variant_id: string;
    tested_at?: string;
    task?: 'detect' | 'classify';
    predictions: Prediction[];
    latency_ms?: number;
    fps_est?: number;
    feedback?: Verdict;
}

interface HistoryItem {
    image_id: string;
    image_file?: string;
    original_name?: string;
    created_at?: string;
    last_tested_at?: string;
    variants?: Record<string, HistoryVariantResult>;
}

interface VariantMeta {
    id: string;
    name: string;
}

interface InferenceResponse extends InferenceResult {
    history_item?: HistoryItem;
}

interface QueueItem {
    key: string;
    file: File;
}

interface BatchProgress {
    active: boolean;
    total: number;
    done: number;
    errors: number;
    message: string;
}

interface PredictionIdentity {
    key: string;
    label: string;
    confidence: number;
}

const IMAGE_EXT_RE = /\.(jpg|jpeg|png|bmp|webp|gif|tif|tiff)$/i;

const isImageFile = (file: File) => {
    if (file.type?.startsWith('image/')) return true;
    return IMAGE_EXT_RE.test(file.name || '');
};

const queueKey = (file: File) => `${file.name}|${file.size}|${file.lastModified}`;

const asPercent = (v: number) => `${Math.max(0, Math.min(100, v)).toFixed(0)}%`;

const formatLatency = (ms?: number) => {
    if (typeof ms !== 'number' || Number.isNaN(ms)) return '—';
    return `${ms.toFixed(1)} ms`;
};

const formatTime = (iso?: string) => {
    if (!iso) return '—';
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return iso;
    return date.toLocaleString();
};

const getPredictionIdentity = (variantResult?: HistoryVariantResult | null): PredictionIdentity => {
    if (!variantResult || !Array.isArray(variantResult.predictions) || variantResult.predictions.length === 0) {
        return { key: 'none', label: 'No prediction', confidence: 0 };
    }

    if (variantResult.task === 'classify') {
        const top = variantResult.predictions[0];
        if (!top) return { key: 'none', label: 'No prediction', confidence: 0 };
        return {
            key: `${top.class_id}:${top.class_name}`,
            label: top.class_name,
            confidence: top.confidence || 0
        };
    }

    const topDetection = [...variantResult.predictions].sort((a, b) => (b.confidence || 0) - (a.confidence || 0))[0];
    if (!topDetection) return { key: 'none', label: 'No detection', confidence: 0 };
    return {
        key: `${topDetection.class_id}:${topDetection.class_name}`,
        label: topDetection.class_name,
        confidence: topDetection.confidence || 0
    };
};

const sortHistoryItems = (items: HistoryItem[]) =>
    [...items].sort((a, b) => String(b.last_tested_at || '').localeCompare(String(a.last_tested_at || '')));

export const ProjectTest = () => {
    const { projectId } = useParams();

    const [variants, setVariants] = useState<VariantMeta[]>([]);
    const [datasetClasses, setDatasetClasses] = useState<string[]>([]);
    const [focusVariantId, setFocusVariantId] = useState<string>('');
    const [history, setHistory] = useState<HistoryItem[]>([]);
    const [historyLoading, setHistoryLoading] = useState(false);
    const [activeHistoryId, setActiveHistoryId] = useState<string | null>(null);
    const [queue, setQueue] = useState<QueueItem[]>([]);
    const [notice, setNotice] = useState<{ kind: 'success' | 'error' | 'info'; text: string } | null>(null);
    const [batchProgress, setBatchProgress] = useState<BatchProgress | null>(null);
    const [singleRunBusy, setSingleRunBusy] = useState<string | null>(null);
    const [feedbackBusy, setFeedbackBusy] = useState(false);

    const imageInputRef = useRef<HTMLInputElement>(null);
    const folderInputRef = useRef<HTMLInputElement>(null);
    const imgRef = useRef<HTMLImageElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);

    const historyImageUrl = useCallback((item: HistoryItem) => {
        if (!projectId) return '';
        return `${ENGINE_HTTP_URL}/projects/${projectId}/inference/history/images/${item.image_id}`;
    }, [projectId]);

    const fetchVariants = useCallback(async () => {
        if (!projectId) return [] as VariantMeta[];
        try {
            const res = await fetch(`${ENGINE_HTTP_URL}/projects/${projectId}/variants`);
            if (!res.ok) return [] as VariantMeta[];
            const data = await res.json();
            const normalized: VariantMeta[] = (Array.isArray(data) ? data : []).map((item: any) => ({
                id: String(item.id),
                name: String(item.name || item.id)
            }));
            setVariants(normalized);
            return normalized;
        } catch (error) {
            console.error(error);
            return [] as VariantMeta[];
        }
    }, [projectId]);

    const fetchDatasetMeta = useCallback(async () => {
        if (!projectId) return;
        try {
            const res = await fetch(`${ENGINE_HTTP_URL}/projects/${projectId}/dataset`);
            if (!res.ok) return;
            const dataset = await res.json();
            const classes = Array.isArray(dataset?.classes) ? dataset.classes : [];
            setDatasetClasses(classes.map((cls: any) => String(cls?.name || cls?.id || '')).filter(Boolean));
        } catch (error) {
            console.error(error);
        }
    }, [projectId]);

    const loadHistory = useCallback(async () => {
        if (!projectId) return;
        setHistoryLoading(true);
        try {
            const res = await fetch(`${ENGINE_HTTP_URL}/projects/${projectId}/inference/history`);
            if (res.ok) {
                const data = await res.json();
                const items = Array.isArray(data) ? data : [];
                setHistory(sortHistoryItems(items));
                if (!activeHistoryId && items.length > 0) {
                    setActiveHistoryId(items[0].image_id);
                }
            }
        } catch (error) {
            console.error(error);
        } finally {
            setHistoryLoading(false);
        }
    }, [projectId, activeHistoryId]);

    useEffect(() => {
        Promise.all([fetchVariants(), fetchDatasetMeta(), loadHistory()]).catch(console.error);
    }, [fetchVariants, fetchDatasetMeta, loadHistory]);

    useEffect(() => {
        if (variants.length === 0) {
            setFocusVariantId('');
            return;
        }
        if (!focusVariantId || !variants.some((variant) => variant.id === focusVariantId)) {
            setFocusVariantId(variants[0].id);
        }
    }, [variants, focusVariantId]);

    useEffect(() => {
        const input = folderInputRef.current;
        if (!input) return;
        input.setAttribute('webkitdirectory', '');
        input.setAttribute('directory', '');
    }, []);

    const upsertHistoryItem = useCallback((incoming: HistoryItem, makeActive = false) => {
        setHistory((prev) => {
            const next = [incoming, ...prev.filter((item) => item.image_id !== incoming.image_id)];
            return sortHistoryItems(next);
        });
        if (makeActive) {
            setActiveHistoryId(incoming.image_id);
        }
    }, []);

    const queueIncomingFiles = useCallback((files: FileList | null) => {
        if (!files || files.length === 0) return;
        const incoming = Array.from(files).filter(isImageFile);
        if (incoming.length === 0) {
            setNotice({ kind: 'error', text: 'Nenhum arquivo de imagem valido foi encontrado.' });
            return;
        }

        setQueue((prev) => {
            const map = new Map(prev.map((item) => [item.key, item]));
            incoming.forEach((file) => {
                const key = queueKey(file);
                map.set(key, { key, file });
            });
            return Array.from(map.values());
        });

        setNotice({ kind: 'info', text: `${incoming.length} imagem(ns) adicionada(s) para validacao em massa.` });
    }, []);

    const inferUploadedFile = useCallback(async (variantId: string, file: File): Promise<InferenceResponse | null> => {
        if (!projectId) return null;
        const formData = new FormData();
        formData.append('file', file);
        formData.append('project_id', projectId);

        const res = await fetch(`${ENGINE_HTTP_URL}/variants/${variantId}/infer`, {
            method: 'POST',
            body: formData
        });

        if (!res.ok) {
            const message = await res.text();
            throw new Error(message || 'Inference failed');
        }

        return await res.json();
    }, [projectId]);

    const inferHistoryImage = useCallback(async (variantId: string, imageId: string): Promise<InferenceResponse | null> => {
        if (!projectId) return null;
        const formData = new FormData();
        formData.append('conf_threshold', '0.25');

        const res = await fetch(`${ENGINE_HTTP_URL}/projects/${projectId}/variants/${variantId}/infer/history/${imageId}`, {
            method: 'POST',
            body: formData
        });

        if (!res.ok) {
            const message = await res.text();
            throw new Error(message || 'Inference failed');
        }

        return await res.json();
    }, [projectId]);

    const applyFeedbackBatch = useCallback(async (imageId: string, updates: { variant_id: string; verdict: Verdict }[]) => {
        if (!projectId || updates.length === 0) return;
        setFeedbackBusy(true);
        try {
            const res = await fetch(`${ENGINE_HTTP_URL}/projects/${projectId}/inference/history/${imageId}/feedback/batch`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ updates })
            });
            if (!res.ok) {
                const message = await res.text();
                throw new Error(message || 'Falha ao salvar feedback em lote');
            }

            setHistory((prev) => prev.map((item) => {
                if (item.image_id !== imageId) return item;
                const variantsMap = { ...(item.variants || {}) };
                updates.forEach((update) => {
                    const target = variantsMap[update.variant_id];
                    if (!target) return;
                    variantsMap[update.variant_id] = { ...target, feedback: update.verdict };
                });
                return { ...item, variants: variantsMap };
            }));
        } finally {
            setFeedbackBusy(false);
        }
    }, [projectId]);

    const runQueuedBatch = useCallback(async () => {
        if (!projectId) return;
        if (queue.length === 0) {
            setNotice({ kind: 'error', text: 'Adicione imagens ou pasta antes de iniciar a validacao.' });
            return;
        }
        if (variants.length === 0) {
            setNotice({ kind: 'error', text: 'Nao existem modelos treinados para validar.' });
            return;
        }

        setNotice(null);
        const variantIds = variants.map((variant) => variant.id);
        const firstVariantId = variantIds[0];
        const collectedImageIds: string[] = [];

        setBatchProgress({
            active: true,
            total: queue.length,
            done: 0,
            errors: 0,
            message: `Iniciando upload e inferencia com ${variants[0].name}...`
        });

        let errorCount = 0;

        for (const queued of queue) {
            try {
                const data = await inferUploadedFile(firstVariantId, queued.file);
                if (data?.history_item) {
                    collectedImageIds.push(data.history_item.image_id);
                    upsertHistoryItem(data.history_item, !activeHistoryId);
                }
            } catch (error) {
                console.error(error);
                errorCount += 1;
            } finally {
                setBatchProgress((prev) => {
                    if (!prev) return prev;
                    return {
                        ...prev,
                        done: prev.done + 1,
                        errors: errorCount,
                        message: `Processando imagens de entrada (${prev.done + 1}/${prev.total})`
                    };
                });
            }
        }

        const uniqueImageIds = Array.from(new Set(collectedImageIds));
        const remainingVariants = variantIds.slice(1);
        const extraOps = remainingVariants.length * uniqueImageIds.length;

        setBatchProgress((prev) => {
            if (!prev) return prev;
            return {
                ...prev,
                total: prev.total + extraOps,
                message: 'Executando validacao cruzada nos demais modelos...'
            };
        });

        for (const variantId of remainingVariants) {
            const variantName = variants.find((variant) => variant.id === variantId)?.name || variantId;
            for (const imageId of uniqueImageIds) {
                try {
                    const data = await inferHistoryImage(variantId, imageId);
                    if (data?.history_item) {
                        upsertHistoryItem(data.history_item, false);
                    }
                } catch (error) {
                    console.error(error);
                    errorCount += 1;
                } finally {
                    setBatchProgress((prev) => {
                        if (!prev) return prev;
                        return {
                            ...prev,
                            done: prev.done + 1,
                            errors: errorCount,
                            message: `${variantName}: ${prev.done + 1}/${prev.total}`
                        };
                    });
                }
            }
        }

        setBatchProgress((prev) => {
            if (!prev) return prev;
            return {
                ...prev,
                active: false,
                message: `Validacao finalizada: ${prev.done}/${prev.total} operacoes.`
            };
        });

        setQueue([]);
        await loadHistory();
        setNotice({
            kind: errorCount > 0 ? 'error' : 'success',
            text: errorCount > 0
                ? `Validacao finalizada com ${errorCount} erro(s).`
                : 'Validacao em massa concluida com sucesso.'
        });
    }, [
        projectId,
        queue,
        variants,
        inferUploadedFile,
        inferHistoryImage,
        upsertHistoryItem,
        loadHistory,
        activeHistoryId
    ]);

    const runMissingCoverage = useCallback(async () => {
        if (!projectId) return;
        if (history.length === 0 || variants.length === 0) {
            setNotice({ kind: 'info', text: 'Nao ha historico suficiente para validacao pendente.' });
            return;
        }

        const byVariant = new Map<string, string[]>();
        history.forEach((item) => {
            variants.forEach((variant) => {
                if (!item.variants?.[variant.id]) {
                    if (!byVariant.has(variant.id)) byVariant.set(variant.id, []);
                    byVariant.get(variant.id)?.push(item.image_id);
                }
            });
        });

        const totalOps = Array.from(byVariant.values()).reduce((sum, ids) => sum + ids.length, 0);
        if (totalOps === 0) {
            setNotice({ kind: 'success', text: 'Todos os modelos ja foram validados nas imagens do historico.' });
            return;
        }

        setBatchProgress({
            active: true,
            total: totalOps,
            done: 0,
            errors: 0,
            message: 'Executando validacao pendente no historico...'
        });

        let errorCount = 0;

        for (const [variantId, imageIds] of byVariant.entries()) {
            const variantName = variants.find((variant) => variant.id === variantId)?.name || variantId;
            for (const imageId of imageIds) {
                try {
                    const data = await inferHistoryImage(variantId, imageId);
                    if (data?.history_item) {
                        upsertHistoryItem(data.history_item, false);
                    }
                } catch (error) {
                    console.error(error);
                    errorCount += 1;
                } finally {
                    setBatchProgress((prev) => {
                        if (!prev) return prev;
                        return {
                            ...prev,
                            done: prev.done + 1,
                            errors: errorCount,
                            message: `${variantName}: ${prev.done + 1}/${prev.total}`
                        };
                    });
                }
            }
        }

        setBatchProgress((prev) => prev ? { ...prev, active: false, message: 'Validacao pendente concluida.' } : prev);
        await loadHistory();
        setNotice({
            kind: errorCount > 0 ? 'error' : 'success',
            text: errorCount > 0
                ? `Validacao pendente concluida com ${errorCount} erro(s).`
                : 'Modelos pendentes validados com sucesso no historico.'
        });
    }, [projectId, history, variants, inferHistoryImage, upsertHistoryItem, loadHistory]);

    const runSingleForActiveImage = useCallback(async (variantId: string) => {
        if (!activeHistoryId) return;
        setSingleRunBusy(variantId);
        try {
            const data = await inferHistoryImage(variantId, activeHistoryId);
            if (data?.history_item) {
                upsertHistoryItem(data.history_item, false);
            }
        } catch (error) {
            console.error(error);
            setNotice({ kind: 'error', text: `Falha ao validar ${variantId} para a imagem selecionada.` });
        } finally {
            setSingleRunBusy(null);
        }
    }, [activeHistoryId, inferHistoryImage, upsertHistoryItem]);

    const activeHistory = useMemo(
        () => history.find((item) => item.image_id === activeHistoryId) || null,
        [history, activeHistoryId]
    );

    const activeImageSrc = useMemo(() => {
        if (!activeHistory) return null;
        return historyImageUrl(activeHistory);
    }, [activeHistory, historyImageUrl]);

    const activeVariantResult = useMemo(() => {
        if (!activeHistory || !focusVariantId) return null;
        return activeHistory.variants?.[focusVariantId] || null;
    }, [activeHistory, focusVariantId]);

    const drawOverlay = useCallback(() => {
        const img = imgRef.current;
        const canvas = canvasRef.current;
        if (!img || !canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        canvas.width = img.width;
        canvas.height = img.height;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const result = activeVariantResult;
        if (!result || result.task === 'classify') {
            return;
        }

        const scaleX = img.width / (img.naturalWidth || img.width || 1);
        const scaleY = img.height / (img.naturalHeight || img.height || 1);

        result.predictions.forEach((prediction) => {
            if (!prediction.box) return;
            const [x1, y1, x2, y2] = prediction.box;
            const sx1 = x1 * scaleX;
            const sy1 = y1 * scaleY;
            const sw = (x2 - x1) * scaleX;
            const sh = (y2 - y1) * scaleY;

            ctx.strokeStyle = '#2fb8ff';
            ctx.lineWidth = 2;
            ctx.strokeRect(sx1, sy1, sw, sh);

            const label = `${prediction.class_name} ${(prediction.confidence * 100).toFixed(1)}%`;
            ctx.font = '12px Sora, sans-serif';
            const textWidth = ctx.measureText(label).width;
            ctx.fillStyle = '#2fb8ff';
            ctx.fillRect(sx1, Math.max(0, sy1 - 20), textWidth + 10, 18);
            ctx.fillStyle = '#0c141a';
            ctx.fillText(label, sx1 + 5, Math.max(11, sy1 - 6));
        });
    }, [activeVariantResult]);

    useEffect(() => {
        drawOverlay();
    }, [drawOverlay, activeImageSrc]);

    const isBinaryProject = datasetClasses.length === 2;

    const modelCoverage = useMemo(() => {
        const map: Record<string, number> = {};
        variants.forEach((variant) => {
            map[variant.id] = 0;
        });
        history.forEach((item) => {
            Object.keys(item.variants || {}).forEach((variantId) => {
                if (map[variantId] === undefined) return;
                map[variantId] += 1;
            });
        });
        return map;
    }, [variants, history]);

    const neverValidatedVariants = useMemo(
        () => variants.filter((variant) => (modelCoverage[variant.id] || 0) === 0),
        [variants, modelCoverage]
    );

    const missingMatrixCount = useMemo(() => {
        if (history.length === 0 || variants.length === 0) return 0;
        let missing = 0;
        history.forEach((item) => {
            variants.forEach((variant) => {
                if (!item.variants?.[variant.id]) {
                    missing += 1;
                }
            });
        });
        return missing;
    }, [history, variants]);

    const activeImageGroups = useMemo(() => {
        if (!activeHistory) return [] as { key: string; label: string; variantIds: string[]; confidence: number }[];
        const groups = new Map<string, { key: string; label: string; variantIds: string[]; confidence: number }>();

        Object.entries(activeHistory.variants || {}).forEach(([variantId, variantResult]) => {
            const identity = getPredictionIdentity(variantResult);
            if (!groups.has(identity.key)) {
                groups.set(identity.key, {
                    key: identity.key,
                    label: identity.label,
                    variantIds: [variantId],
                    confidence: identity.confidence
                });
            } else {
                const current = groups.get(identity.key)!;
                current.variantIds.push(variantId);
                current.confidence = Math.max(current.confidence, identity.confidence);
            }
        });

        return Array.from(groups.values())
            .filter((group) => group.key !== 'none')
            .sort((a, b) => b.variantIds.length - a.variantIds.length);
    }, [activeHistory]);

    const applyPropagationFeedback = useCallback(async (variantId: string, verdict: Verdict) => {
        if (!activeHistory) return;

        const sourceResult = activeHistory.variants?.[variantId];
        if (!sourceResult) return;

        const sourceIdentity = getPredictionIdentity(sourceResult);
        const updatesMap = new Map<string, Verdict>();

        Object.entries(activeHistory.variants || {}).forEach(([candidateVariantId, candidateResult]) => {
            const candidateIdentity = getPredictionIdentity(candidateResult);
            if (candidateIdentity.key === sourceIdentity.key) {
                updatesMap.set(candidateVariantId, verdict);
            }
        });

        if (isBinaryProject && verdict) {
            Object.entries(activeHistory.variants || {}).forEach(([candidateVariantId, candidateResult]) => {
                const candidateIdentity = getPredictionIdentity(candidateResult);
                if (candidateIdentity.key === sourceIdentity.key) return;
                if (candidateIdentity.key === 'none') return;
                updatesMap.set(candidateVariantId, verdict === 'correct' ? 'incorrect' : 'correct');
            });
        }

        const updates = Array.from(updatesMap.entries()).map(([targetVariantId, targetVerdict]) => ({
            variant_id: targetVariantId,
            verdict: targetVerdict
        }));

        try {
            await applyFeedbackBatch(activeHistory.image_id, updates);
        } catch (error) {
            console.error(error);
            setNotice({ kind: 'error', text: 'Falha ao propagar feedback da inferencia.' });
        }
    }, [activeHistory, applyFeedbackBatch, isBinaryProject]);

    const markClassAsCorrect = useCallback(async (classKey: string) => {
        if (!activeHistory) return;
        if (classKey === 'none') return;

        const updatesMap = new Map<string, Verdict>();
        Object.entries(activeHistory.variants || {}).forEach(([variantId, variantResult]) => {
            const identity = getPredictionIdentity(variantResult);
            if (identity.key === classKey) {
                updatesMap.set(variantId, 'correct');
            }
        });

        if (isBinaryProject) {
            Object.entries(activeHistory.variants || {}).forEach(([variantId, variantResult]) => {
                const identity = getPredictionIdentity(variantResult);
                if (identity.key !== classKey && identity.key !== 'none') {
                    updatesMap.set(variantId, 'incorrect');
                }
            });
        }

        const updates = Array.from(updatesMap.entries()).map(([variantId, verdict]) => ({
            variant_id: variantId,
            verdict
        }));

        try {
            await applyFeedbackBatch(activeHistory.image_id, updates);
        } catch (error) {
            console.error(error);
            setNotice({ kind: 'error', text: 'Falha ao aplicar validacao por classe.' });
        }
    }, [activeHistory, applyFeedbackBatch, isBinaryProject]);

    const historyWithCoverage = useMemo(() => {
        const totalModels = Math.max(variants.length, 1);
        return history.map((item) => {
            const testedModels = Object.keys(item.variants || {}).length;
            return {
                ...item,
                testedModels,
                coveragePct: (testedModels / totalModels) * 100
            };
        });
    }, [history, variants.length]);

    const activeCoverage = useMemo(() => {
        if (!activeHistory) return { tested: 0, total: variants.length };
        return {
            tested: Object.keys(activeHistory.variants || {}).length,
            total: variants.length
        };
    }, [activeHistory, variants.length]);

    const activeProgressPct = useMemo(() => {
        if (activeCoverage.total === 0) return 0;
        return (activeCoverage.tested / activeCoverage.total) * 100;
    }, [activeCoverage]);

    return (
        <div className="project-test animate-fade-in w-full max-w-none space-y-6 pb-16">
            <header className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
                <div className="flex items-start gap-3">
                    <div className="rounded-xl border border-accent-primary/30 bg-accent-primary/10 p-2">
                        <ScanSearch size={20} className="text-accent-primary" />
                    </div>
                    <div>
                        <h2 className="text-3xl font-bold">Mass Validation Playground</h2>
                        <p className="text-secondary">Valide varias imagens e pastas contra todos os modelos treinados.</p>
                    </div>
                </div>
                <StatusPill
                    status={batchProgress?.active ? 'running' : 'success'}
                    label={batchProgress?.active ? 'VALIDATION RUNNING' : 'READY'}
                    pulsing={Boolean(batchProgress?.active)}
                />
            </header>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
                <Card className="border border-graphite-700/70 bg-graphite-900/55 p-4">
                    <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.32em] text-secondary">
                        <span>Models</span>
                        <Target size={14} />
                    </div>
                    <div className="mt-2 text-2xl font-mono">{variants.length}</div>
                    <div className="mt-1 text-xs text-secondary">{neverValidatedVariants.length} sem validacao</div>
                </Card>

                <Card className="border border-graphite-700/70 bg-graphite-900/55 p-4">
                    <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.32em] text-secondary">
                        <span>History Images</span>
                        <Camera size={14} />
                    </div>
                    <div className="mt-2 text-2xl font-mono">{history.length}</div>
                    <div className="mt-1 text-xs text-secondary">Imagens no repositorio de validacao</div>
                </Card>

                <Card className="border border-graphite-700/70 bg-graphite-900/55 p-4">
                    <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.32em] text-secondary">
                        <span>Missing Matrix</span>
                        <Clock3 size={14} />
                    </div>
                    <div className="mt-2 text-2xl font-mono">{missingMatrixCount}</div>
                    <div className="mt-1 text-xs text-secondary">Combinacoes imagem x modelo pendentes</div>
                </Card>

                <Card className={`border p-4 ${isBinaryProject ? 'border-accent-success/35 bg-accent-success/5' : 'border-graphite-700/70 bg-graphite-900/55'}`}>
                    <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.32em] text-secondary">
                        <span>Class Rules</span>
                        <CheckCircle2 size={14} />
                    </div>
                    <div className="mt-2 text-sm font-semibold">{isBinaryProject ? 'Binary propagation ON' : 'Multi-class mode'}</div>
                    <div className="mt-1 text-xs text-secondary">
                        {isBinaryProject
                            ? 'Acerto/erro propaga automaticamente para a classe oposta.'
                            : 'Propagacao automatica da classe oposta desativada (3+ classes).'}
                    </div>
                </Card>
            </div>

            {notice && (
                <Card className={`border p-3 ${
                    notice.kind === 'error'
                        ? 'border-accent-error/45 bg-accent-error/10'
                        : notice.kind === 'success'
                            ? 'border-accent-success/45 bg-accent-success/10'
                            : 'border-accent-primary/35 bg-accent-primary/10'
                }`}>
                    <div className="flex items-center justify-between gap-3 text-sm">
                        <div>{notice.text}</div>
                        <Button size="sm" variant="ghost" onClick={() => setNotice(null)}>
                            <XCircle size={13} className="mr-1" />
                            Close
                        </Button>
                    </div>
                </Card>
            )}

            {batchProgress && (
                <Card className={`border ${batchProgress.active ? 'border-accent-warning/45 bg-accent-warning/10' : 'border-graphite-700/70 bg-graphite-900/55'}`}>
                    <div className="flex items-center justify-between gap-3">
                        <div className="flex items-center gap-2 text-xs uppercase tracking-[0.3em] text-secondary">
                            {batchProgress.active ? <Loader2 size={14} className="animate-spin text-accent-warning" /> : <CheckCircle2 size={14} className="text-accent-success" />}
                            Validation Progress
                        </div>
                        <div className="text-xs text-secondary">
                            {batchProgress.done}/{batchProgress.total} · errors {batchProgress.errors}
                        </div>
                    </div>
                    <div className="mt-2 text-sm text-secondary">{batchProgress.message}</div>
                    <div className="mt-3 h-2 overflow-hidden rounded bg-graphite-700">
                        <div
                            className={`h-full transition-all duration-300 ${batchProgress.active ? 'bg-accent-warning' : 'bg-accent-success'}`}
                            style={{ width: `${batchProgress.total > 0 ? (batchProgress.done / batchProgress.total) * 100 : 0}%` }}
                        />
                    </div>
                </Card>
            )}

            <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
                <div className="space-y-6 xl:col-span-4">
                    <Card className="space-y-4">
                        <div className="flex items-center gap-2 text-xs uppercase tracking-[0.32em] text-secondary">
                            <Upload size={14} />
                            Input Queue
                        </div>

                        <div className="grid grid-cols-2 gap-2">
                            <Button
                                variant="secondary"
                                onClick={() => imageInputRef.current?.click()}
                                disabled={Boolean(batchProgress?.active)}
                            >
                                <ImagePlus size={14} className="mr-1" />
                                Add Images
                            </Button>
                            <Button
                                variant="secondary"
                                onClick={() => folderInputRef.current?.click()}
                                disabled={Boolean(batchProgress?.active)}
                            >
                                <FolderOpen size={14} className="mr-1" />
                                Add Folder
                            </Button>
                        </div>

                        <input
                            ref={imageInputRef}
                            type="file"
                            multiple
                            accept="image/*"
                            className="hidden"
                            onChange={(event) => {
                                queueIncomingFiles(event.target.files);
                                event.target.value = '';
                            }}
                        />
                        <input
                            ref={folderInputRef}
                            type="file"
                            multiple
                            accept="image/*"
                            className="hidden"
                            onChange={(event) => {
                                queueIncomingFiles(event.target.files);
                                event.target.value = '';
                            }}
                        />

                        <div className="rounded-lg border border-graphite-700/70 bg-graphite-900/60 px-3 py-2">
                            <div className="flex items-center justify-between text-xs uppercase tracking-[0.24em] text-secondary">
                                <span>Queued Images</span>
                                <span>{queue.length}</span>
                            </div>
                            {queue.length === 0 ? (
                                <div className="mt-2 text-xs text-secondary">Nenhuma imagem adicionada.</div>
                            ) : (
                                <div className="mt-2 max-h-40 space-y-1 overflow-y-auto pr-1 text-xs">
                                    {queue.slice(0, 40).map((item) => (
                                        <div key={item.key} className="flex items-center justify-between rounded border border-graphite-700/50 bg-graphite-950/60 px-2 py-1">
                                            <span className="truncate pr-2">{item.file.webkitRelativePath || item.file.name}</span>
                                            <button
                                                type="button"
                                                className="text-accent-error hover:text-accent-error/80"
                                                onClick={() => setQueue((prev) => prev.filter((q) => q.key !== item.key))}
                                            >
                                                ×
                                            </button>
                                        </div>
                                    ))}
                                    {queue.length > 40 && (
                                        <div className="text-secondary">+{queue.length - 40} itens ocultos...</div>
                                    )}
                                </div>
                            )}
                        </div>

                        <div className="grid grid-cols-1 gap-2">
                            <Button
                                variant="primary"
                                onClick={runQueuedBatch}
                                disabled={Boolean(batchProgress?.active) || queue.length === 0 || variants.length === 0}
                            >
                                <PlayCircle size={14} className="mr-1" />
                                Run All Images x All Models
                            </Button>
                            <Button
                                variant="ghost"
                                onClick={runMissingCoverage}
                                disabled={Boolean(batchProgress?.active) || missingMatrixCount === 0}
                            >
                                <RefreshCcw size={14} className="mr-1" />
                                Validate Missing History Matrix
                            </Button>
                            <Button
                                variant="ghost"
                                onClick={() => setQueue([])}
                                disabled={Boolean(batchProgress?.active) || queue.length === 0}
                            >
                                Clear Queue
                            </Button>
                        </div>
                    </Card>

                    <Card className="space-y-3">
                        <div className="flex items-center justify-between">
                            <div className="text-xs uppercase tracking-[0.32em] text-secondary">Model Coverage</div>
                            <StatusPill status={neverValidatedVariants.length > 0 ? 'warning' : 'success'} label={`${neverValidatedVariants.length} NEW`} />
                        </div>

                        <div className="max-h-[280px] space-y-2 overflow-y-auto pr-1">
                            {variants.map((variant) => {
                                const testedCount = modelCoverage[variant.id] || 0;
                                const coveragePct = history.length > 0 ? (testedCount / history.length) * 100 : 0;
                                const isNew = testedCount === 0;
                                return (
                                    <div key={variant.id} className={`rounded-lg border px-3 py-2 ${isNew ? 'border-accent-warning/50 bg-accent-warning/10' : 'border-graphite-700/70 bg-graphite-900/60'}`}>
                                        <div className="flex items-center justify-between gap-2">
                                            <div className="truncate text-sm font-semibold">{variant.name}</div>
                                            <div className="text-[10px] uppercase tracking-[0.24em] text-secondary">
                                                {testedCount}/{history.length || 0}
                                            </div>
                                        </div>
                                        <div className="mt-2 h-1.5 overflow-hidden rounded bg-graphite-700">
                                            <div className={`h-full ${isNew ? 'bg-accent-warning' : 'bg-accent-primary'}`} style={{ width: `${coveragePct}%` }} />
                                        </div>
                                        <div className="mt-1 text-[11px] text-secondary">
                                            {isNew ? 'Novo modelo sem validacao.' : `Cobertura ${asPercent(coveragePct)}`}
                                        </div>
                                    </div>
                                );
                            })}
                            {variants.length === 0 && (
                                <div className="text-sm text-secondary">Nenhum modelo treinado encontrado.</div>
                            )}
                        </div>
                    </Card>

                    <Card className="space-y-3">
                        <div className="flex items-center justify-between">
                            <div className="text-xs uppercase tracking-[0.32em] text-secondary">Validation History</div>
                            <Button size="sm" variant="ghost" onClick={loadHistory} disabled={historyLoading || Boolean(batchProgress?.active)}>
                                <RefreshCcw size={13} className="mr-1" />
                                {historyLoading ? 'Loading...' : 'Refresh'}
                            </Button>
                        </div>

                        {historyWithCoverage.length === 0 ? (
                            <div className="text-sm text-secondary">Nenhuma imagem validada ainda.</div>
                        ) : (
                            <div className="grid grid-cols-3 gap-2 max-h-64 overflow-y-auto pr-1">
                                {historyWithCoverage.map((item) => (
                                    <button
                                        key={item.image_id}
                                        type="button"
                                        onClick={() => setActiveHistoryId(item.image_id)}
                                        className={`rounded border p-1 text-left transition-all ${
                                            activeHistoryId === item.image_id
                                                ? 'border-accent-primary bg-accent-primary/10'
                                                : 'border-graphite-700/70 bg-graphite-900/60'
                                        }`}
                                    >
                                        <img
                                            src={historyImageUrl(item)}
                                            alt={item.original_name || item.image_id}
                                            className="h-16 w-full rounded object-cover"
                                        />
                                        <div className="mt-1 h-1 w-full overflow-hidden rounded bg-graphite-700">
                                            <div className="h-full bg-accent-primary" style={{ width: `${item.coveragePct}%` }} />
                                        </div>
                                        <div className="mt-1 text-[10px] uppercase tracking-[0.18em] text-secondary">
                                            {item.testedModels}/{variants.length || 0}
                                        </div>
                                    </button>
                                ))}
                            </div>
                        )}
                    </Card>
                </div>

                <div className="space-y-6 xl:col-span-8">
                    <Card>
                        <div className="mb-3 flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                            <div>
                                <h3 className="text-xl font-semibold">Active Image Validation</h3>
                                <p className="text-xs uppercase tracking-[0.24em] text-secondary">
                                    {activeHistory ? `${activeCoverage.tested}/${activeCoverage.total} models tested` : 'Select an image from history'}
                                </p>
                            </div>
                            <div className="w-full max-w-sm">
                                <Select
                                    label="Preview model"
                                    options={variants.map((variant) => ({ value: variant.id, label: variant.name }))}
                                    value={focusVariantId}
                                    onChange={(event) => setFocusVariantId(event.target.value)}
                                />
                            </div>
                        </div>

                        <div className="mb-4 h-2 overflow-hidden rounded bg-graphite-700">
                            <div className="h-full bg-accent-primary" style={{ width: `${activeProgressPct}%` }} />
                        </div>

                        <div className="grid grid-cols-1 gap-4 xl:grid-cols-12">
                            <div className="xl:col-span-7">
                                <div className="relative min-h-[380px] rounded-xl border border-graphite-700/70 bg-black/35 p-3">
                                    {!activeImageSrc && (
                                        <div className="absolute inset-0 flex items-center justify-center text-center text-secondary">
                                            <div>
                                                <Camera size={40} className="mx-auto mb-3 opacity-70" />
                                                <div>Selecione uma imagem no historico para validar resultados.</div>
                                            </div>
                                        </div>
                                    )}

                                    {activeImageSrc && (
                                        <div className="relative flex h-full min-h-[360px] items-center justify-center">
                                            <img
                                                ref={imgRef}
                                                src={activeImageSrc}
                                                alt={activeHistory?.original_name || activeHistory?.image_id}
                                                className="max-h-[500px] max-w-full rounded object-contain"
                                                onLoad={drawOverlay}
                                            />
                                            <canvas ref={canvasRef} className="pointer-events-none absolute inset-0" />
                                        </div>
                                    )}
                                </div>
                            </div>

                            <div className="space-y-3 xl:col-span-5">
                                <div className="rounded-xl border border-graphite-700/70 bg-graphite-900/55 p-3">
                                    <div className="text-xs uppercase tracking-[0.24em] text-secondary">Selected Image</div>
                                    <div className="mt-1 truncate text-sm font-semibold">{activeHistory?.original_name || activeHistory?.image_id || '—'}</div>
                                    <div className="mt-2 text-xs text-secondary">Last tested: {formatTime(activeHistory?.last_tested_at)}</div>
                                    <div className="mt-1 text-xs text-secondary">Model preview: {variants.find((variant) => variant.id === focusVariantId)?.name || '—'}</div>
                                </div>

                                <div className="rounded-xl border border-graphite-700/70 bg-graphite-900/55 p-3">
                                    <div className="mb-2 text-xs uppercase tracking-[0.24em] text-secondary">Prediction Overview</div>
                                    {activeVariantResult ? (
                                        <>
                                            <div className="text-sm font-semibold">Top: {getPredictionIdentity(activeVariantResult).label}</div>
                                            <div className="mt-1 text-xs text-secondary">Confidence {(getPredictionIdentity(activeVariantResult).confidence * 100).toFixed(1)}%</div>
                                            <div className="mt-1 text-xs text-secondary">Latency {formatLatency(activeVariantResult.latency_ms)}</div>
                                            <div className="mt-3 space-y-1 text-xs">
                                                {activeVariantResult.predictions.slice(0, 5).map((prediction, index) => (
                                                    <div key={`${prediction.class_id}-${index}`} className="flex items-center justify-between rounded border border-graphite-700/70 bg-graphite-950/50 px-2 py-1">
                                                        <span>{prediction.class_name}</span>
                                                        <span className="font-mono">{(prediction.confidence * 100).toFixed(1)}%</span>
                                                    </div>
                                                ))}
                                            </div>
                                        </>
                                    ) : (
                                        <div className="text-xs text-secondary">Este modelo ainda nao foi executado para a imagem selecionada.</div>
                                    )}
                                </div>

                                <div className="rounded-xl border border-graphite-700/70 bg-graphite-900/55 p-3">
                                    <div className="mb-2 text-xs uppercase tracking-[0.24em] text-secondary">Class Consensus</div>
                                    {activeImageGroups.length === 0 ? (
                                        <div className="text-xs text-secondary">Sem inferencias para consolidar.</div>
                                    ) : (
                                        <div className="flex flex-wrap gap-2">
                                            {activeImageGroups.map((group) => (
                                                <button
                                                    key={group.key}
                                                    type="button"
                                                    className="tab-button tab-button-active"
                                                    onClick={() => markClassAsCorrect(group.key)}
                                                    disabled={feedbackBusy}
                                                    title="Marcar esta classe como correta para os modelos que previram este resultado"
                                                >
                                                    {group.label} ({group.variantIds.length})
                                                </button>
                                            ))}
                                        </div>
                                    )}
                                    {isBinaryProject && (
                                        <div className="mt-2 text-xs text-secondary">
                                            Projeto binario: ao marcar uma classe correta, a outra classe e marcada como incorreta automaticamente.
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    </Card>

                    <Card>
                        <div className="mb-4 flex items-center justify-between">
                            <h3 className="text-lg font-semibold">Model-by-Model Validation</h3>
                            {feedbackBusy && (
                                <span className="inline-flex items-center gap-2 text-xs text-secondary">
                                    <Loader2 size={12} className="animate-spin" />
                                    Saving feedback...
                                </span>
                            )}
                        </div>

                        {!activeHistory ? (
                            <div className="rounded border border-graphite-700/70 bg-graphite-900/55 p-4 text-sm text-secondary">
                                Selecione uma imagem no historico para avaliar cada modelo.
                            </div>
                        ) : (
                            <div className="overflow-x-auto">
                                <table className="min-w-full text-sm">
                                    <thead>
                                        <tr className="border-b border-graphite-700/70 text-xs uppercase tracking-[0.22em] text-secondary">
                                            <th className="px-3 py-2 text-left">Model</th>
                                            <th className="px-3 py-2 text-left">Prediction</th>
                                            <th className="px-3 py-2 text-left">Latency</th>
                                            <th className="px-3 py-2 text-left">Feedback</th>
                                            <th className="px-3 py-2 text-right">Action</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {variants.map((variant) => {
                                            const variantResult = activeHistory.variants?.[variant.id] || null;
                                            const identity = getPredictionIdentity(variantResult);
                                            const feedback = variantResult?.feedback || null;
                                            const neverValidated = (modelCoverage[variant.id] || 0) === 0;
                                            const runningSingle = singleRunBusy === variant.id;
                                            return (
                                                <tr key={variant.id} className="border-b border-graphite-700/30">
                                                    <td className="px-3 py-3">
                                                        <div className="flex items-center gap-2">
                                                            <button
                                                                type="button"
                                                                className={`text-left ${focusVariantId === variant.id ? 'text-accent-primary' : 'text-bone'}`}
                                                                onClick={() => setFocusVariantId(variant.id)}
                                                            >
                                                                {variant.name}
                                                            </button>
                                                            {neverValidated && (
                                                                <span className="rounded border border-accent-warning/45 bg-accent-warning/10 px-2 py-0.5 text-[10px] uppercase tracking-[0.2em] text-accent-warning">
                                                                    New
                                                                </span>
                                                            )}
                                                        </div>
                                                    </td>
                                                    <td className="px-3 py-3">
                                                        {variantResult ? (
                                                            <div className="text-xs">
                                                                <div className="font-semibold">{identity.label}</div>
                                                                <div className="text-secondary">{(identity.confidence * 100).toFixed(1)}%</div>
                                                            </div>
                                                        ) : (
                                                            <span className="text-xs text-secondary">Not tested</span>
                                                        )}
                                                    </td>
                                                    <td className="px-3 py-3 text-xs text-secondary">{formatLatency(variantResult?.latency_ms)}</td>
                                                    <td className="px-3 py-3">
                                                        {variantResult ? (
                                                            <div className="flex items-center gap-2">
                                                                <Button
                                                                    size="sm"
                                                                    variant="secondary"
                                                                    className={feedback === 'correct' ? 'border-accent-success text-accent-success' : ''}
                                                                    onClick={() => applyPropagationFeedback(variant.id, feedback === 'correct' ? null : 'correct')}
                                                                    disabled={feedbackBusy}
                                                                >
                                                                    Correct
                                                                </Button>
                                                                <Button
                                                                    size="sm"
                                                                    variant="secondary"
                                                                    className={feedback === 'incorrect' ? 'border-accent-error text-accent-error' : ''}
                                                                    onClick={() => applyPropagationFeedback(variant.id, feedback === 'incorrect' ? null : 'incorrect')}
                                                                    disabled={feedbackBusy}
                                                                >
                                                                    Wrong
                                                                </Button>
                                                            </div>
                                                        ) : (
                                                            <span className="text-xs text-secondary">Run to review</span>
                                                        )}
                                                    </td>
                                                    <td className="px-3 py-3 text-right">
                                                        {variantResult ? (
                                                            <span className="text-xs text-secondary">{formatTime(variantResult.tested_at)}</span>
                                                        ) : (
                                                            <Button
                                                                size="sm"
                                                                variant="ghost"
                                                                onClick={() => runSingleForActiveImage(variant.id)}
                                                                disabled={Boolean(batchProgress?.active) || runningSingle}
                                                            >
                                                                {runningSingle ? (
                                                                    <>
                                                                        <Loader2 size={12} className="mr-1 animate-spin" />
                                                                        Running
                                                                    </>
                                                                ) : (
                                                                    'Run'
                                                                )}
                                                            </Button>
                                                        )}
                                                    </td>
                                                </tr>
                                            );
                                        })}
                                    </tbody>
                                </table>
                            </div>
                        )}

                        <div className="mt-4 rounded border border-graphite-700/70 bg-graphite-900/60 px-3 py-2 text-xs text-secondary">
                            <div className="font-semibold text-bone mb-1">Regras de propagacao ativas</div>
                            <div>1) Ao marcar um resultado como correto/incorreto, todos os modelos que previram a mesma classe para esta imagem recebem o mesmo feedback.</div>
                            <div>2) Em projeto com 2 classes, a classe oposta recebe feedback inverso automaticamente.</div>
                            <div>3) Em projeto com 3 ou mais classes, a classe oposta nao e marcada automaticamente.</div>
                        </div>
                    </Card>
                </div>
            </div>
        </div>
    );
};
