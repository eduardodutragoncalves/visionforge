import { useEffect, useMemo, useRef, useState } from 'react';
import { useParams } from 'react-router-dom';
import clsx from 'clsx';
import { FolderOpen, Image, Film, Plus, X } from 'lucide-react';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { Select } from '../components/ui/Select';
import { ENGINE_HTTP_URL, ENGINE_WS_URL } from '../lib/engine';

interface DatasetClass {
    id: number;
    name: string;
    color?: string;
}

interface BoundingBox {
    class_id: number;
    x: number;
    y: number;
    w: number;
    h: number;
}

interface DatasetImage {
    id: string;
    path: string;
    split?: string;
    labels: number[];
    boxes: BoundingBox[];
}

interface Dataset {
    id: string;
    project_id: string;
    root_path: string;
    task: 'classification' | 'detection';
    format: 'yolo' | 'coco' | 'pascal' | 'tfrecord';
    classes: DatasetClass[];
    images: DatasetImage[];
}

const API_BASE_URL = ENGINE_HTTP_URL;
type SaveStatus = 'idle' | 'saving' | 'pending' | 'saved';
type PendingSave =
    | { type: 'labels'; imageId: string; labels: number[]; updatedAt: number }
    | { type: 'boxes'; imageId: string; boxes: BoundingBox[]; updatedAt: number };

const toFileUrl = (path: string) => {
    const normalized = path.replace(/\\/g, '/');
    if (normalized.startsWith('/')) {
        return `file://${normalized}`;
    }
    return `file:///${normalized}`;
};

export const ProjectDataset = () => {
    const { projectId } = useParams();
    const [dataset, setDataset] = useState<Dataset | null>(null);
    const [images, setImages] = useState<DatasetImage[]>([]);
    const [loading, setLoading] = useState(true);
    const [ingesting, setIngesting] = useState(false);
    const [scanProgress, setScanProgress] = useState<any>(null);
    const [validationReport, setValidationReport] = useState<any>(null);
    const [isValidating, setIsValidating] = useState(false);
    const [newClassName, setNewClassName] = useState('');
    const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
    const [activeImageId, setActiveImageId] = useState<string | null>(null);
    const [previewImageId, setPreviewImageId] = useState<string | null>(null);
    const [activeClassId, setActiveClassId] = useState<number | null>(null);
    const [classificationTab, setClassificationTab] = useState<'unlabeled' | 'labeled'>('unlabeled');
    const [thumbSize, setThumbSize] = useState<'sm' | 'lg' | 'xl'>('lg');
    const [frameInterval, setFrameInterval] = useState(30);
    const [draftBox, setDraftBox] = useState<BoundingBox | null>(null);
    const [isDrawing, setIsDrawing] = useState(false);
    const [imageAspect, setImageAspect] = useState<Record<string, 'portrait' | 'landscape' | 'square'>>({});
    const [saveStatus, setSaveStatus] = useState<SaveStatus>('idle');
    const [pendingCount, setPendingCount] = useState(0);
    const [saveError, setSaveError] = useState<string | null>(null);
    const thumbSettings = useMemo(() => {
        switch (thumbSize) {
            case 'sm':
                return { min: 120, height: 88 };
            case 'xl':
                return { min: 210, height: 160 };
            default:
                return { min: 160, height: 120 };
        }
    }, [thumbSize]);
    const imageRef = useRef<HTMLImageElement | null>(null);
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const pendingRef = useRef<PendingSave[]>([]);
    const flushRef = useRef(false);
    const classes = dataset?.classes ?? [];

    const classMap = useMemo(() => new Map(classes.map((cls) => [cls.id, cls.name])), [classes]);
    const saveStatusLabel = useMemo(() => {
        if (saveStatus === 'saving') return 'Saving...';
        if (saveStatus === 'pending') return pendingCount > 0 ? `Pending (${pendingCount})` : 'Pending';
        if (saveStatus === 'saved') return 'All changes saved';
        return 'Auto-save ready';
    }, [saveStatus, pendingCount]);
    const getImageSrc = (image: DatasetImage) => {
        if (projectId) {
            return `${API_BASE_URL}/projects/${projectId}/dataset/images/${image.id}/file`;
        }
        return toFileUrl(image.path);
    };

    const selectedImage = useMemo(
        () => images.find((image) => image.id === activeImageId) || null,
        [images, activeImageId]
    );
    const previewImage = useMemo(
        () => images.find((image) => image.id === previewImageId) || null,
        [images, previewImageId]
    );
    const labeledImages = useMemo(
        () => images.filter((image) => image.labels.length > 0),
        [images]
    );
    const unlabeledImages = useMemo(
        () => images.filter((image) => image.labels.length === 0),
        [images]
    );
    const labeledCounts = useMemo(() => {
        const counts = new Map<number, number>();
        classes.forEach((cls) => counts.set(cls.id, 0));
        labeledImages.forEach((image) => {
            image.labels.forEach((id) => counts.set(id, (counts.get(id) || 0) + 1));
        });
        return counts;
    }, [classes, labeledImages]);
    const visibleImages = classificationTab === 'labeled' ? labeledImages : unlabeledImages;

    const handleThumbLoad = (imageId: string, event: React.SyntheticEvent<HTMLImageElement>) => {
        const target = event.currentTarget;
        if (!target.naturalWidth || !target.naturalHeight) return;
        const ratio = target.naturalWidth / target.naturalHeight;
        const nextAspect: 'portrait' | 'landscape' | 'square' =
            ratio > 1.15 ? 'landscape' : ratio < 0.85 ? 'portrait' : 'square';
        setImageAspect((prev) => (prev[imageId] === nextAspect ? prev : { ...prev, [imageId]: nextAspect }));
    };

    const getPendingKey = (id: string) => `visionforge.dataset.pending.${id}`;

    const readPendingSaves = (id: string): PendingSave[] => {
        if (typeof window === 'undefined') return [];
        try {
            const raw = window.localStorage.getItem(getPendingKey(id));
            if (!raw) return [];
            const parsed = JSON.parse(raw);
            if (!Array.isArray(parsed)) return [];
            return parsed.filter(
                (item) =>
                    item &&
                    (item.type === 'labels' || item.type === 'boxes') &&
                    typeof item.imageId === 'string'
            );
        } catch {
            return [];
        }
    };

    const writePendingSaves = (id: string, items: PendingSave[]) => {
        if (typeof window === 'undefined') return;
        const key = getPendingKey(id);
        if (items.length === 0) {
            window.localStorage.removeItem(key);
            return;
        }
        window.localStorage.setItem(key, JSON.stringify(items));
    };

    const applyPendingToImages = (base: DatasetImage[], pending: PendingSave[]) => {
        if (pending.length === 0) return base;
        const byId = new Map(base.map((image) => [image.id, image]));
        pending.forEach((item) => {
            const image = byId.get(item.imageId);
            if (!image) return;
            if (item.type === 'labels') {
                byId.set(item.imageId, { ...image, labels: item.labels });
            } else {
                byId.set(item.imageId, { ...image, boxes: item.boxes });
            }
        });
        return Array.from(byId.values());
    };

    const updatePendingStore = (next: PendingSave[]) => {
        pendingRef.current = next;
        if (projectId) {
            writePendingSaves(projectId, next);
        }
        setPendingCount(next.length);
    };

    const flushPending = async () => {
        if (!projectId || flushRef.current) return;
        if (pendingRef.current.length === 0) {
            setSaveStatus('saved');
            setSaveError(null);
            return;
        }
        flushRef.current = true;
        setSaveStatus('saving');
        setSaveError(null);
        while (pendingRef.current.length > 0) {
            const item = pendingRef.current[0];
            try {
                let res: Response;
                if (item.type === 'labels') {
                    res = await fetch(
                        `${API_BASE_URL}/projects/${projectId}/dataset/images/${item.imageId}/labels`,
                        {
                            method: 'PUT',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ labels: item.labels })
                        }
                    );
                } else {
                    res = await fetch(
                        `${API_BASE_URL}/projects/${projectId}/dataset/images/${item.imageId}/boxes`,
                        {
                            method: 'PUT',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ boxes: item.boxes })
                        }
                    );
                }
                if (!res.ok) {
                    throw new Error(`HTTP ${res.status}`);
                }
                const remaining = pendingRef.current.slice(1);
                updatePendingStore(remaining);
            } catch {
                setSaveStatus('pending');
                setSaveError('Auto-save paused. Will retry on next change.');
                flushRef.current = false;
                return;
            }
        }
        flushRef.current = false;
        setSaveStatus('saved');
        setSaveError(null);
    };

    const enqueuePendingBatch = (items: PendingSave[]) => {
        if (!projectId || items.length === 0) return;
        const next = pendingRef.current.slice();
        items.forEach((item) => {
            const existingIndex = next.findIndex(
                (entry) => entry.type === item.type && entry.imageId === item.imageId
            );
            if (existingIndex >= 0) {
                next.splice(existingIndex, 1);
            }
            next.push(item);
        });
        updatePendingStore(next);
        setSaveStatus('pending');
        setSaveError(null);
        void flushPending();
    };

    const enqueuePending = (item: PendingSave) => {
        enqueuePendingBatch([item]);
    };

    const fetchDataset = async () => {
        if (!projectId) return;
        setLoading(true);
        try {
            const res = await fetch(`${API_BASE_URL}/projects/${projectId}/dataset`);
            if (res.ok) {
                const data = await res.json();
                setDataset(data);
            }
            const imagesRes = await fetch(`${API_BASE_URL}/projects/${projectId}/dataset/images?limit=500`);
            if (imagesRes.ok) {
                const nextImages: DatasetImage[] = await imagesRes.json();
                if (pendingRef.current.length === 0) {
                    const pending = readPendingSaves(projectId);
                    pendingRef.current = pending;
                    setPendingCount(pending.length);
                    if (pending.length > 0) {
                        setSaveStatus('pending');
                    }
                }
                setImages(applyPendingToImages(nextImages, pendingRef.current));
                if (pendingRef.current.length > 0) {
                    void flushPending();
                } else {
                    setSaveStatus('saved');
                }
            }
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        pendingRef.current = [];
        setPendingCount(0);
        setSaveStatus('idle');
        setSaveError(null);
    }, [projectId]);

    useEffect(() => {
        fetchDataset();
    }, [projectId]);

    useEffect(() => {
        if (!activeImageId && images.length > 0) {
            setActiveImageId(images[0].id);
        }
    }, [images, activeImageId]);

    useEffect(() => {
        const handleMessage = (event: MessageEvent) => {
            try {
                const msg = JSON.parse(event.data);
                if (msg.event === 'dataset/scan_progress') {
                    setScanProgress(msg.data);
                } else if (msg.event === 'dataset/validate_progress') {
                    setScanProgress(msg.data);
                } else if (msg.event === 'dataset/validated') {
                    setValidationReport(msg.data);
                    setIsValidating(false);
                }
            } catch {
                // ignore
            }
        };

        const ws = new WebSocket(ENGINE_WS_URL);
        ws.onmessage = handleMessage;

        return () => ws.close();
    }, []);

    const updateDatasetConfig = async (updates: Partial<Pick<Dataset, 'format'>>) => {
        if (!projectId) return;
        const res = await fetch(`${API_BASE_URL}/projects/${projectId}/dataset`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(updates)
        });
        if (res.ok) {
            setDataset(await res.json());
        }
    };

    const handleAddClass = async () => {
        if (!projectId || !dataset) return;
        const name = newClassName.trim();
        if (!name) return;
        const res = await fetch(`${API_BASE_URL}/projects/${projectId}/dataset/classes`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                classes: [...dataset.classes, { name }]
            })
        });
        if (res.ok) {
            setDataset(await res.json());
            setNewClassName('');
        }
    };

    const handleRemoveClass = async (classId: number) => {
        if (!projectId || !dataset) return;
        const res = await fetch(`${API_BASE_URL}/projects/${projectId}/dataset/classes`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                classes: dataset.classes.filter((c) => c.id !== classId)
            })
        });
        if (res.ok) {
            setDataset(await res.json());
        }
    };

    const handleIngestDirectories = async () => {
        if (!projectId) return;
        const path = await window.ipcRenderer?.openDirectory?.();
        if (!path) return;
        setIngesting(true);
        await fetch(`${API_BASE_URL}/projects/${projectId}/dataset/ingest/directories`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ paths: [path] })
        });
        await fetchDataset();
        setIngesting(false);
    };

    const handleIngestFiles = async () => {
        if (!projectId) return;
        const paths = await window.ipcRenderer?.openFiles?.();
        if (!paths || paths.length === 0) return;
        setIngesting(true);
        await fetch(`${API_BASE_URL}/projects/${projectId}/dataset/ingest/files`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ paths })
        });
        await fetchDataset();
        setIngesting(false);
    };

    const handleIngestVideo = async () => {
        if (!projectId) return;
        const path = await window.ipcRenderer?.openVideo?.();
        if (!path) return;
        setIngesting(true);
        await fetch(`${API_BASE_URL}/projects/${projectId}/dataset/ingest/video`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path, frame_interval: frameInterval })
        });
        await fetchDataset();
        setIngesting(false);
    };

    const handleValidate = async () => {
        if (!projectId) return;
        setIsValidating(true);
        setValidationReport(null);
        await fetch(`${API_BASE_URL}/projects/${projectId}/dataset/validate`, { method: 'POST' });
    };

    const toggleSelected = (imageId: string) => {
        setSelectedIds((prev) => {
            const next = new Set(prev);
            if (next.has(imageId)) {
                next.delete(imageId);
            } else {
                next.add(imageId);
            }
            return next;
        });
    };

    const handleAssignClass = async (classId: number) => {
        if (!projectId || selectedIds.size === 0) return;
        const ids = Array.from(selectedIds);
        setImages((prev) =>
            prev.map((image) =>
                ids.includes(image.id) ? { ...image, labels: [classId] } : image
            )
        );
        const updatedAt = Date.now();
        enqueuePendingBatch(
            ids.map((id) => ({ type: 'labels', imageId: id, labels: [classId], updatedAt }))
        );
        setSelectedIds(new Set());
    };

    const handleClearLabels = async () => {
        if (!projectId || selectedIds.size === 0) return;
        const ids = Array.from(selectedIds);
        setImages((prev) =>
            prev.map((image) =>
                ids.includes(image.id) ? { ...image, labels: [] } : image
            )
        );
        const updatedAt = Date.now();
        enqueuePendingBatch(
            ids.map((id) => ({ type: 'labels', imageId: id, labels: [], updatedAt }))
        );
        setSelectedIds(new Set());
    };

    const handleDeleteImages = async (imageIds: string[]) => {
        if (!projectId || imageIds.length === 0) return;
        const confirmed = window.confirm(`Delete ${imageIds.length} image(s) from this project?`);
        if (!confirmed) return;
        const res = await fetch(`${API_BASE_URL}/projects/${projectId}/dataset/images/delete`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_ids: imageIds })
        });
        if (!res.ok) return;
        setImages((prev) => prev.filter((image) => !imageIds.includes(image.id)));
        if (pendingRef.current.length > 0) {
            const remaining = pendingRef.current.filter((item) => !imageIds.includes(item.imageId));
            updatePendingStore(remaining);
            if (remaining.length === 0 && !flushRef.current) {
                setSaveStatus('saved');
            }
        }
        if (activeImageId && imageIds.includes(activeImageId)) {
            setActiveImageId(null);
        }
        setSelectedIds((prev) => {
            const next = new Set(prev);
            imageIds.forEach((id) => next.delete(id));
            return next;
        });
        if (previewImageId && imageIds.includes(previewImageId)) {
            setPreviewImageId(null);
        }
    };

    const handleAssignClassSingle = async (imageId: string, classId: number) => {
        if (!projectId) return;
        setImages((prev) =>
            prev.map((image) =>
                image.id === imageId ? { ...image, labels: [classId] } : image
            )
        );
        enqueuePending({ type: 'labels', imageId, labels: [classId], updatedAt: Date.now() });
    };

    const handleClearLabelsSingle = async (imageId: string) => {
        if (!projectId) return;
        setImages((prev) =>
            prev.map((image) => (image.id === imageId ? { ...image, labels: [] } : image))
        );
        enqueuePending({ type: 'labels', imageId, labels: [], updatedAt: Date.now() });
    };

    const handleDeleteImage = async (imageId: string) => {
        await handleDeleteImages([imageId]);
    };

    const updateBoxes = async (imageId: string, boxes: BoundingBox[]) => {
        if (!projectId) return;
        setImages((prev) =>
            prev.map((image) => (image.id === imageId ? { ...image, boxes } : image))
        );
        enqueuePending({ type: 'boxes', imageId, boxes, updatedAt: Date.now() });
    };

    const drawBoxes = (boxes: BoundingBox[], draft?: BoundingBox | null) => {
        const canvas = canvasRef.current;
        const img = imageRef.current;
        if (!canvas || !img) return;
        const rect = img.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const allBoxes = draft ? [...boxes, draft] : boxes;
        allBoxes.forEach((box) => {
            const x = box.x * rect.width;
            const y = box.y * rect.height;
            const w = box.w * rect.width;
            const h = box.h * rect.height;
            ctx.strokeStyle = '#2fb8ff';
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, w, h);
        });
    };

    useEffect(() => {
        if (selectedImage) {
            drawBoxes(selectedImage.boxes, draftBox);
        }
    }, [selectedImage, draftBox]);

    const getRelativePoint = (event: React.MouseEvent<HTMLDivElement, MouseEvent>) => {
        const img = imageRef.current;
        if (!img) return null;
        const rect = img.getBoundingClientRect();
        const x = Math.min(Math.max(event.clientX - rect.left, 0), rect.width);
        const y = Math.min(Math.max(event.clientY - rect.top, 0), rect.height);
        return { x, y, width: rect.width, height: rect.height };
    };

    const handleMouseDown = (event: React.MouseEvent<HTMLDivElement, MouseEvent>) => {
        if (!selectedImage || activeClassId === null) return;
        const point = getRelativePoint(event);
        if (!point) return;
        setIsDrawing(true);
        setDraftBox({
            class_id: activeClassId,
            x: point.x / point.width,
            y: point.y / point.height,
            w: 0,
            h: 0
        });
    };

    const handleMouseMove = (event: React.MouseEvent<HTMLDivElement, MouseEvent>) => {
        if (!isDrawing || !draftBox) return;
        const point = getRelativePoint(event);
        if (!point) return;
        const startX = draftBox.x * point.width;
        const startY = draftBox.y * point.height;
        const width = point.x - startX;
        const height = point.y - startY;
        setDraftBox({
            ...draftBox,
            w: Math.abs(width) / point.width,
            h: Math.abs(height) / point.height,
            x: Math.min(startX, point.x) / point.width,
            y: Math.min(startY, point.y) / point.height
        });
    };

    const handleMouseUp = async () => {
        if (!selectedImage || !draftBox) {
            setIsDrawing(false);
            return;
        }
        const isTiny = draftBox.w < 0.01 || draftBox.h < 0.01;
        setIsDrawing(false);
        if (isTiny) {
            setDraftBox(null);
            return;
        }
        const nextBoxes = [...selectedImage.boxes, draftBox];
        setDraftBox(null);
        await updateBoxes(selectedImage.id, nextBoxes);
    };

    const handleRemoveBox = async (index: number) => {
        if (!selectedImage) return;
        const nextBoxes = selectedImage.boxes.filter((_, idx) => idx !== index);
        await updateBoxes(selectedImage.id, nextBoxes);
    };

    const totalSelected = selectedIds.size;

    useEffect(() => {
        if (!previewImageId) return;
        const handleKey = (event: KeyboardEvent) => {
            if (event.key === 'Escape') {
                setPreviewImageId(null);
            }
        };
        window.addEventListener('keydown', handleKey);
        return () => window.removeEventListener('keydown', handleKey);
    }, [previewImageId]);

    useEffect(() => {
        setSelectedIds(new Set());
    }, [classificationTab]);

    return (
        <div className="animate-fade-in space-y-6">
            <header className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
                <div>
                    <div className="text-xs uppercase tracking-[0.6em] text-haze">Dataset</div>
                    <h2 className="text-3xl font-semibold">Build and Label</h2>
                    <p className="text-secondary text-sm mt-2">
                        Import directories, files, or video frames, then label for {dataset?.task || 'classification'}.
                    </p>
                </div>
                <div className="flex items-center gap-3">
                    <Button variant="secondary" onClick={handleValidate} disabled={isValidating}>
                        {isValidating ? 'Validating...' : 'Validate dataset'}
                    </Button>
                </div>
            </header>

            <div className="grid gap-6 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,0.8fr)]">
                <Card className="space-y-4">
                    <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                        <div>
                            <h3 className="text-lg font-semibold">Ingestion</h3>
                            <p className="text-secondary text-sm">Bring data into the project workspace.</p>
                        </div>
                        <div className="flex flex-wrap gap-2">
                            <Button variant="secondary" onClick={handleIngestDirectories} disabled={ingesting}>
                                <FolderOpen size={16} />
                                Add folder
                            </Button>
                            <Button variant="secondary" onClick={handleIngestFiles} disabled={ingesting}>
                                <Image size={16} />
                                Add files
                            </Button>
                            <Button variant="secondary" onClick={handleIngestVideo} disabled={ingesting}>
                                <Film size={16} />
                                Add video
                            </Button>
                        </div>
                    </div>

                    <div className="grid gap-3 md:grid-cols-[1fr_auto]">
                        <Input
                            label="Frame interval (video)"
                            type="number"
                            value={frameInterval}
                            onChange={(e) => setFrameInterval(Number(e.target.value) || 1)}
                        />
                        <Select
                            label="Dataset format"
                            options={[
                                { value: 'yolo', label: 'YOLO' },
                                { value: 'coco', label: 'COCO' },
                                { value: 'pascal', label: 'Pascal VOC' },
                                { value: 'tfrecord', label: 'TFRecord' }
                            ]}
                            value={dataset?.format || 'yolo'}
                            onChange={(e) => updateDatasetConfig({ format: e.target.value as Dataset['format'] })}
                        />
                    </div>

                    <div className="flex flex-wrap gap-4 text-xs uppercase tracking-[0.3em] text-haze">
                        <span>Images: {images.length}</span>
                        <span>Classes: {dataset?.classes.length || 0}</span>
                    </div>
                </Card>

                <Card className="space-y-4">
                    <div className="flex items-center justify-between">
                        <div>
                            <h3 className="text-lg font-semibold">Classes</h3>
                            <p className="text-secondary text-sm">Create labels used during training.</p>
                        </div>
                    </div>

                    <div className="flex gap-2">
                        <Input
                            placeholder="Add class name"
                            value={newClassName}
                            onChange={(e) => setNewClassName(e.target.value)}
                        />
                        <Button variant="primary" onClick={handleAddClass}>
                            <Plus size={16} />
                        </Button>
                    </div>

                    <div className="flex flex-wrap gap-2">
                        {classes.map((cls) => (
                            <span
                                key={cls.id}
                                className="flex items-center gap-2 border-2 border-graphite-700 px-3 py-1 text-xs uppercase tracking-[0.3em]"
                            >
                                <span className="h-2 w-2 rounded-full" style={{ background: cls.color || '#2fb8ff' }} />
                                {cls.name}
                                <button onClick={() => handleRemoveClass(cls.id)} aria-label="Remove class">
                                    <X size={12} />
                                </button>
                            </span>
                        ))}
                    </div>
                </Card>
            </div>

            {isValidating && (
                <Card>
                    <h3>Scanning & validating...</h3>
                    <div className="mt-4">
                        <div className="text-sm text-secondary mb-2">
                            Processed: {scanProgress?.processed || scanProgress?.count || 0} files
                        </div>
                        <div className="h-2 bg-graphite-700 rounded overflow-hidden">
                            <div
                                className="h-full bg-accent-primary transition-all duration-300"
                                style={{ width: `${scanProgress?.progress || 0}%` }}
                            />
                        </div>
                    </div>
                </Card>
            )}

            {validationReport && (
                <div className="grid grid-cols-2 gap-6">
                    <Card glass>
                        <h3 className="mb-4 text-accent-success">Validation passed</h3>
                        <div className="text-4xl font-bold">
                            {validationReport.valid_count}{' '}
                            <span className="text-base text-secondary font-normal">images</span>
                        </div>
                        <div className="mt-4 text-xs text-secondary font-mono break-all">
                            Hash: {validationReport.dataset_hash}
                        </div>
                    </Card>
                    <Card glass>
                        <h3 className="mb-4 text-accent-error">Issues found</h3>
                        <div className="text-4xl font-bold">
                            {validationReport.corrupted_count}{' '}
                            <span className="text-base text-secondary font-normal">corrupted</span>
                        </div>
                        <div className="mt-4 max-h-32 overflow-y-auto">
                            {validationReport.corrupted_files.map((f: string, i: number) => (
                                <div key={i} className="text-xs text-error font-mono">{f}</div>
                            ))}
                        </div>
                    </Card>
                </div>
            )}

            {loading ? (
                <Card>
                    <div className="text-secondary">Loading dataset...</div>
                </Card>
            ) : (
                <>
            {dataset && dataset.task === 'classification' && (
                        <Card className="space-y-4 flex flex-col h-[calc(100vh-320px)] max-h-[calc(100vh-320px)] min-h-[480px] overflow-hidden">
                            <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                                <div>
                                    <h3 className="text-lg font-semibold">Classification labeling</h3>
                                    <p className="text-secondary text-sm">
                                        Select images and assign a class in batch.
                                    </p>
                                </div>
                                <div className="flex flex-wrap items-center gap-2 text-xs uppercase tracking-[0.3em] text-haze">
                                    Selected: {totalSelected}
                                    <span className="text-secondary">{saveStatusLabel}</span>
                                </div>
                            </div>
                            {saveError && (
                                <div className="text-xs text-error">{saveError}</div>
                            )}

                            <div className="flex flex-wrap gap-2">
                                {classes.map((cls) => (
                                    <Button
                                        key={cls.id}
                                        size="sm"
                                        variant="secondary"
                                        onClick={() => handleAssignClass(cls.id)}
                                    >
                                        {cls.name}
                                    </Button>
                                ))}
                                <Button size="sm" variant="ghost" onClick={handleClearLabels}>
                                    Clear labels
                                </Button>
                                <Button
                                    size="sm"
                                    variant="secondary"
                                    className="border-accent-error/60 text-accent-error hover:text-white"
                                    onClick={() => handleDeleteImages(Array.from(selectedIds))}
                                    disabled={selectedIds.size === 0}
                                >
                                    Delete selected
                                </Button>
                            </div>

                            <div className="flex flex-wrap items-center justify-between gap-3">
                                <div className="flex flex-wrap gap-2">
                                    <button
                                        type="button"
                                        className={clsx(
                                            'tab-button',
                                            classificationTab === 'unlabeled' && 'tab-button-active'
                                        )}
                                        onClick={() => setClassificationTab('unlabeled')}
                                    >
                                        Unlabeled ({unlabeledImages.length})
                                    </button>
                                    <button
                                        type="button"
                                        className={clsx(
                                            'tab-button',
                                            classificationTab === 'labeled' && 'tab-button-active'
                                        )}
                                        onClick={() => setClassificationTab('labeled')}
                                    >
                                        Labeled ({labeledImages.length})
                                    </button>
                                </div>
                                <div className="flex items-center gap-2 text-xs uppercase tracking-[0.3em] text-secondary">
                                    <span>Thumbs</span>
                                    <div className="flex items-center gap-2">
                                        {(['sm', 'lg', 'xl'] as const).map((size) => (
                                            <button
                                                key={size}
                                                type="button"
                                                className={clsx('thumb-toggle', thumbSize === size && 'thumb-toggle-active')}
                                                onClick={() => setThumbSize(size)}
                                            >
                                                {size.toUpperCase()}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            </div>
                            {classificationTab === 'labeled' && (
                                <div className="flex flex-wrap gap-2 text-[11px] uppercase tracking-[0.3em] text-secondary">
                                    {classes.map((cls) => (
                                        <span
                                            key={`count-${cls.id}`}
                                            className="border border-graphite-700/70 bg-graphite-900 px-3 py-1"
                                        >
                                            {cls.name}: {labeledCounts.get(cls.id) || 0}
                                        </span>
                                    ))}
                                </div>
                            )}

                            <div
                                className={clsx('image-grid', `image-grid-${thumbSize}`, 'flex-1 min-h-0')}
                                style={
                                    {
                                        gridTemplateColumns: `repeat(auto-fill, minmax(${thumbSettings.min}px, 1fr))`,
                                        '--thumb-height': `${thumbSettings.height}px`
                                    } as React.CSSProperties
                                }
                            >
                                {visibleImages.map((image) => (
                                    <button
                                        key={image.id}
                                        type="button"
                                        onClick={(event) => {
                                            if (event.detail > 1) return;
                                            toggleSelected(image.id);
                                        }}
                                        onDoubleClick={(event) => {
                                            event.preventDefault();
                                            event.stopPropagation();
                                            setPreviewImageId(image.id);
                                        }}
                                        className={clsx(
                                            'image-card',
                                            imageAspect[image.id] ? `image-card-${imageAspect[image.id]}` : null,
                                            selectedIds.has(image.id) ? 'image-card-selected' : null
                                        )}
                                    >
                                        <img
                                            src={getImageSrc(image)}
                                            alt={image.id}
                                            className="thumb-image"
                                            onLoad={(event) => handleThumbLoad(image.id, event)}
                                        />
                                        <div className="mt-2 text-xs text-secondary">
                                            {image.labels.length > 0
                                                ? `Label: ${image.labels.map((id) => classMap.get(id) || id).join(', ')}`
                                                : 'Unlabeled'}
                                        </div>
                                    </button>
                                ))}
                            </div>
                        </Card>
                    )}

                    {previewImage && dataset?.task === 'classification' && (
                        <div className="modal-backdrop" role="dialog" aria-modal="true">
                            <div className="modal-card">
                                <div className="modal-header">
                                    <div>
                                        <div className="text-xs uppercase tracking-[0.4em] text-haze">Preview</div>
                                        <div className="text-lg font-semibold">{previewImage.id}</div>
                                    </div>
                                    <button
                                        type="button"
                                        className="window-button"
                                        onClick={() => setPreviewImageId(null)}
                                        aria-label="Close"
                                    >
                                        <X size={14} />
                                    </button>
                                </div>
                                <img
                                    src={getImageSrc(previewImage)}
                                    alt={previewImage.id}
                                    className="modal-image"
                                />
                                <div className="modal-labels">
                                    {classes.map((cls) => (
                                        <Button
                                            key={cls.id}
                                            size="sm"
                                            variant="secondary"
                                            onClick={() => handleAssignClassSingle(previewImage.id, cls.id)}
                                            className={clsx(
                                                previewImage.labels.includes(cls.id) && 'border-accent-primary text-white'
                                            )}
                                        >
                                            {cls.name}
                                        </Button>
                                    ))}
                                    <Button
                                        size="sm"
                                        variant="ghost"
                                        onClick={() => handleClearLabelsSingle(previewImage.id)}
                                    >
                                        Clear label
                                    </Button>
                                </div>
                                <div className="modal-actions">
                                    <div className="text-xs text-secondary">
                                        {previewImage.labels.length > 0
                                            ? `Label: ${previewImage.labels.map((id) => classMap.get(id) || id).join(', ')}`
                                            : 'Unlabeled'}
                                    </div>
                                    <Button
                                        size="sm"
                                        variant="secondary"
                                        className="border-accent-error/60 text-accent-error hover:text-white"
                                        onClick={() => handleDeleteImage(previewImage.id)}
                                    >
                                        Delete image
                                    </Button>
                                </div>
                            </div>
                        </div>
                    )}

                            {dataset && dataset.task === 'detection' && (
                        <Card className="space-y-4">
                            <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                                <div>
                                    <h3 className="text-lg font-semibold">Detection labeling</h3>
                                    <p className="text-secondary text-sm">
                                        Select an image, choose a class, and draw bounding boxes.
                                    </p>
                                </div>
                                <div className="flex flex-wrap items-center gap-3">
                                    <span className="text-xs uppercase tracking-[0.3em] text-secondary">
                                        {saveStatusLabel}
                                    </span>
                                    <Select
                                        label="Active class"
                                        options={classes.map((cls) => ({ value: String(cls.id), label: cls.name }))}
                                        value={activeClassId !== null ? String(activeClassId) : ''}
                                        onChange={(e) => setActiveClassId(Number(e.target.value))}
                                    />
                                </div>
                            </div>
                            {saveError && (
                                <div className="text-xs text-error">{saveError}</div>
                            )}

                            <div className="grid gap-4 lg:grid-cols-[220px_minmax(0,1fr)]">
                                <div className="space-y-2">
                                    {images.map((image) => (
                                        <button
                                            key={image.id}
                                            type="button"
                                            onClick={() => setActiveImageId(image.id)}
                                            className={`flex w-full items-center gap-2 border-2 px-2 py-1 text-left ${activeImageId === image.id ? 'border-accent-primary' : 'border-graphite-700'}`}
                                        >
                                            <span className="text-xs text-secondary">{image.id}</span>
                                        </button>
                                    ))}
                                </div>

                                <div className="space-y-4">
                                    <div
                                        className="relative border-2 border-graphite-700 bg-graphite-900 p-2"
                                        onMouseDown={handleMouseDown}
                                        onMouseMove={handleMouseMove}
                                        onMouseUp={handleMouseUp}
                                    >
                                        {selectedImage ? (
                                            <>
                                                <img
                                                    ref={imageRef}
                                                    src={getImageSrc(selectedImage)}
                                                    alt={selectedImage.id}
                                                    className="max-h-[520px] w-full object-contain"
                                                    onLoad={() => drawBoxes(selectedImage.boxes, draftBox)}
                                                />
                                                <canvas ref={canvasRef} className="absolute inset-0 pointer-events-none" />
                                            </>
                                        ) : (
                                            <div className="text-secondary text-sm">Select an image to start labeling.</div>
                                        )}
                                    </div>

                                    {selectedImage && (
                                        <div className="space-y-2">
                                            <div className="text-xs uppercase tracking-[0.3em] text-haze">Boxes</div>
                                            {selectedImage.boxes.map((box, index) => (
                                                <div key={`${selectedImage.id}-${index}`} className="flex items-center justify-between border-2 border-graphite-700 px-2 py-1 text-xs">
                                                    <span>Class {box.class_id}</span>
                                                    <button onClick={() => handleRemoveBox(index)} className="text-error">Remove</button>
                                                </div>
                                            ))}
                                            {selectedImage.boxes.length === 0 && (
                                                <div className="text-xs text-secondary">No boxes yet.</div>
                                            )}
                                        </div>
                                    )}
                                </div>
                            </div>
                        </Card>
                    )}
                </>
            )}
        </div>
    );
};
