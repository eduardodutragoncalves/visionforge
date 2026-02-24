import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import clsx from 'clsx';
import { Copy, FolderOpen, Image, Film, Loader2, Plus, X, GripVertical, Search, ChevronLeft, ChevronRight, Sparkles } from 'lucide-react';
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

type ResizeHandle = 'nw' | 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w';

interface ResizeDragState {
    boxIndex: number;
    handle: ResizeHandle;
    sourceBox: BoundingBox;
}

interface PreclassifyPrediction {
    class_id: number;
    class_name: string;
    confidence: number;
}

interface DatasetImage {
    id: string;
    path: string;
    split?: string;
    labels: number[];
    boxes: BoundingBox[];
    detection_done?: boolean;
    is_augmented?: boolean;
    augmentation_source_id?: string | null;
}

type ImageOrientation = 'landscape' | 'portrait' | 'square';

interface ImageMetadata {
    width?: number;
    height?: number;
    megapixels?: number;
    orientation?: ImageOrientation;
    fileSizeBytes?: number | null;
}

interface Dataset {
    id: string;
    project_id: string;
    root_path: string;
    task: 'classification' | 'detection';
    format?: 'yolo' | 'coco' | 'pascal' | 'tfrecord';
    classes: DatasetClass[];
    images: DatasetImage[];
}

type ClassificationAugmentedFilter = 'all' | 'original' | 'augmented';

interface ClassificationAugmentationConfig {
    target_scope: 'all' | 'labeled' | 'unlabeled';
    copies_per_image: number;
    output_size: number;
    apply_percent: number;
    horizontal_flip_percent: number;
    vertical_flip_percent: number;
    hsv_h_percent: number;
    hsv_s_percent: number;
    hsv_v_percent: number;
    erasing_percent: number;
    auto_augment: '' | 'randaugment' | 'augmix' | 'autoaugment';
    force_color_jitter: boolean;
}

interface ClassificationAugmentationProgress {
    status: 'idle' | 'starting' | 'running' | 'completed';
    processed: number;
    total: number;
    percent: number;
    added: number;
}

interface AutoLabelModelOption {
    id: string;
    name: string;
    base_model?: string;
    run_id?: string;
    created_at?: string;
}

interface AutoLabelSuggestion {
    image_id: string;
    variant_id: string;
    run_id?: string;
    predicted_class_id: number;
    predicted_class_name?: string;
    confidence: number;
    status: 'pending' | 'accepted' | 'rejected';
    created_at?: string;
    updated_at?: string;
}

interface AutoLabelProgressState {
    status: 'idle' | 'starting' | 'running' | 'completed' | 'error';
    processed: number;
    total: number;
    progress: number;
    message: string;
    suggested?: number;
    removed?: number;
    errors?: number;
}

interface DuplicateImageEntry {
    id?: string | null;
    path: string;
    size: number;
    content_hash: string;
    labels_count: number;
    boxes_count: number;
    detection_done: boolean;
    annotated: boolean;
}

interface DuplicateGroup {
    content_hash: string;
    count: number;
    images: DuplicateImageEntry[];
}

interface ValidationReport {
    valid_count: number;
    corrupted_count: number;
    corrupted_files: string[];
    dataset_hash: string;
    duplicate_groups_count: number;
    duplicate_files_count: number;
    duplicate_groups: DuplicateGroup[];
}

type DuplicateResolutionAction = 'ignore' | 'delete_keep_one' | 'delete_all';

const API_BASE_URL = ENGINE_HTTP_URL;
const IMAGE_PAGE_SIZE = 500;
const CLASSIFICATION_INITIAL_RENDER = 120;
const CLASSIFICATION_RENDER_CHUNK = 80;
const DETECTION_INITIAL_RENDER = 90;
const DETECTION_RENDER_CHUNK = 60;
const PRECLASSIFY_ENABLED = false;
type SaveStatus = 'idle' | 'saving' | 'pending' | 'saved';
type PendingSave =
    | { type: 'labels'; imageId: string; labels: number[]; updatedAt: number }
    | { type: 'boxes'; imageId: string; boxes: BoundingBox[]; updatedAt: number }
    | { type: 'detection_done'; imageId: string; detection_done: boolean; updatedAt: number };
type LabelChange = { imageId: string; prevLabels: number[]; nextLabels: number[] };

const isEditableTarget = (target: EventTarget | null) => {
    if (!target || !(target instanceof HTMLElement)) return false;
    const tag = target.tagName;
    return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || target.isContentEditable;
};

const getShortcutDigit = (event: KeyboardEvent): number | null => {
    const code = String(event.code || '');
    const codeMatch = code.match(/^(Digit|Numpad)([1-9])$/);
    if (codeMatch) {
        return Number(codeMatch[2]);
    }
    const key = String(event.key || '').trim();
    if (/^[1-9]$/.test(key)) {
        return Number(key);
    }
    return null;
};

const labelsEqual = (a: number[], b: number[]) =>
    a.length === b.length && a.every((value, index) => value === b[index]);

const toFileUrl = (path: string) => {
    const normalized = path.replace(/\\/g, '/');
    if (normalized.startsWith('/')) {
        return `file://${normalized}`;
    }
    return `file:///${normalized}`;
};

const clampPercent = (value: number) => Math.max(0, Math.min(100, value));

const formatFileSize = (bytes?: number | null) => {
    if (typeof bytes !== 'number' || !Number.isFinite(bytes) || bytes <= 0) return 'Unknown';
    const units = ['B', 'KB', 'MB', 'GB'];
    let value = bytes;
    let unitIndex = 0;
    while (value >= 1024 && unitIndex < units.length - 1) {
        value /= 1024;
        unitIndex += 1;
    }
    const precision = value >= 10 || unitIndex === 0 ? 0 : 1;
    return `${value.toFixed(precision)} ${units[unitIndex]}`;
};

const getOrientationFromSize = (width: number, height: number): ImageOrientation => {
    if (!width || !height) return 'square';
    const ratio = width / height;
    if (ratio > 1.05) return 'landscape';
    if (ratio < 0.95) return 'portrait';
    return 'square';
};

const getOrientationLabel = (orientation?: ImageOrientation) => {
    if (!orientation) return 'Unknown';
    if (orientation === 'landscape') return 'Landscape';
    if (orientation === 'portrait') return 'Portrait';
    return 'Square';
};

const getResolutionTier = (width?: number, height?: number) => {
    if (!width || !height) return null;
    const long = Math.max(width, height);
    const short = Math.min(width, height);
    if (long >= 7680 && short >= 4320) return '8K';
    if (long >= 3840 && short >= 2160) return '4K';
    if (long >= 2560 && short >= 1440) return 'QHD';
    if (long >= 1920 && short >= 1080) return 'Full HD';
    if (long >= 1280 && short >= 720) return 'HD';
    return 'SD';
};

const getSuggestionLabel = (suggestion?: AutoLabelSuggestion | null) => {
    if (!suggestion) return '';
    const suggestedName = String(suggestion.predicted_class_name || '').trim();
    if (suggestedName) return suggestedName;
    return `Class ${suggestion.predicted_class_id}`;
};

const getFileNameFromPath = (path: string) => {
    const parts = path.split(/[/\\]/);
    return parts[parts.length - 1] || path;
};

const compareDuplicateCandidates = (a: DuplicateImageEntry, b: DuplicateImageEntry) => {
    const scoreA = (a.annotated ? 1000 : 0) + a.labels_count + a.boxes_count + (a.detection_done ? 1 : 0);
    const scoreB = (b.annotated ? 1000 : 0) + b.labels_count + b.boxes_count + (b.detection_done ? 1 : 0);
    if (scoreA !== scoreB) return scoreB - scoreA;
    return a.path.localeCompare(b.path);
};

const formatElapsedSeconds = (seconds: number) => {
    const safe = Math.max(0, Math.trunc(seconds));
    const minutes = Math.floor(safe / 60);
    const remaining = safe % 60;
    if (minutes <= 0) return `${remaining}s`;
    return `${minutes}m ${String(remaining).padStart(2, '0')}s`;
};

const readErrorMessage = async (response: Response, fallback: string) => {
    try {
        const payload = await response.json();
        const detail = String(payload?.detail || '').trim();
        return detail || fallback;
    } catch {
        const text = await response.text().catch(() => '');
        return String(text || fallback).trim() || fallback;
    }
};

export const ProjectDataset = () => {
    const navigate = useNavigate();
    const { projectId } = useParams();
    const [dataset, setDataset] = useState<Dataset | null>(null);
    const [images, setImages] = useState<DatasetImage[]>([]);
    const [loading, setLoading] = useState(true);
    const [datasetError, setDatasetError] = useState<string | null>(null);
    const [ingesting, setIngesting] = useState(false);
    const [scanProgress, setScanProgress] = useState<any>(null);
    const [validationReport, setValidationReport] = useState<ValidationReport | null>(null);
    const [isValidating, setIsValidating] = useState(false);
    const [validationElapsedSec, setValidationElapsedSec] = useState(0);
    const [validationLastUpdateAt, setValidationLastUpdateAt] = useState<number | null>(null);
    const [isDuplicateModalOpen, setIsDuplicateModalOpen] = useState(false);
    const [duplicateResolutionError, setDuplicateResolutionError] = useState<string | null>(null);
    const [isResolvingDuplicates, setIsResolvingDuplicates] = useState(false);
    const [newClassName, setNewClassName] = useState('');
    const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
    const [activeImageId, setActiveImageId] = useState<string | null>(null);
    const [previewImageId, setPreviewImageId] = useState<string | null>(null);
    const [activeClassId, setActiveClassId] = useState<number | null>(null);
    const [classificationTab, setClassificationTab] = useState<'unlabeled' | 'labeled'>('unlabeled');
    const [detectionTab, setDetectionTab] = useState<'unlabeled' | 'labeled'>('unlabeled');
    const [labeledClassFilter, setLabeledClassFilter] = useState<number | null>(null);
    const [unlabeledSuggestionFilter, setUnlabeledSuggestionFilter] = useState<number | null>(null);
    const [classificationAugmentedFilter, setClassificationAugmentedFilter] = useState<ClassificationAugmentedFilter>('all');
    const [detectionLabeledClassFilters, setDetectionLabeledClassFilters] = useState<number[]>([]);
    const [thumbSize, setThumbSize] = useState<'sm' | 'lg' | 'xl'>('lg');
    const [detectionSidebarWidth, setDetectionSidebarWidth] = useState(380);
    const [isResizingDetectionSidebar, setIsResizingDetectionSidebar] = useState(false);
    const [frameInterval, setFrameInterval] = useState(30);
    const [draftBox, setDraftBox] = useState<BoundingBox | null>(null);
    const [isDrawing, setIsDrawing] = useState(false);
    const [editingBoxIndex, setEditingBoxIndex] = useState<number | null>(null);
    const [editingBoxDraft, setEditingBoxDraft] = useState<BoundingBox | null>(null);
    const [resizeDrag, setResizeDrag] = useState<ResizeDragState | null>(null);
    const [imageAspect, setImageAspect] = useState<Record<string, 'portrait' | 'landscape' | 'square'>>({});
    const [imageMetadataById, setImageMetadataById] = useState<Record<string, ImageMetadata>>({});
    const [saveStatus, setSaveStatus] = useState<SaveStatus>('idle');
    const [pendingCount, setPendingCount] = useState(0);
    const [saveError, setSaveError] = useState<string | null>(null);
    const [imageOffset, setImageOffset] = useState(0);
    const [imageTotal, setImageTotal] = useState(0);
    const [classificationRenderCount, setClassificationRenderCount] = useState(CLASSIFICATION_INITIAL_RENDER);
    const [detectionRenderCount, setDetectionRenderCount] = useState(DETECTION_INITIAL_RENDER);
    const [isPreclassifying, setIsPreclassifying] = useState(false);
    const [preclassifyModel, setPreclassifyModel] = useState('yolov8n-cls.pt');
    const [preclassifyTopK, setPreclassifyTopK] = useState(3);
    const [preclassifyError, setPreclassifyError] = useState<string | null>(null);
    const [preclassifyResults, setPreclassifyResults] = useState<Record<string, PreclassifyPrediction[]>>({});
    const [autoLabelModels, setAutoLabelModels] = useState<AutoLabelModelOption[]>([]);
    const [autoLabelModelId, setAutoLabelModelId] = useState('');
    const [autoLabelSuggestionsByImage, setAutoLabelSuggestionsByImage] = useState<Record<string, AutoLabelSuggestion>>({});
    const [autoLabelLoadingModels, setAutoLabelLoadingModels] = useState(false);
    const [autoLabelRunning, setAutoLabelRunning] = useState(false);
    const [autoLabelMinConfidence, setAutoLabelMinConfidence] = useState(0.7);
    const [autoLabelError, setAutoLabelError] = useState<string | null>(null);
    const [autoLabelNotice, setAutoLabelNotice] = useState<string | null>(null);
    const [autoLabelProgress, setAutoLabelProgress] = useState<AutoLabelProgressState>({
        status: 'idle',
        processed: 0,
        total: 0,
        progress: 0,
        message: ''
    });
    const [isAugmentationModalOpen, setIsAugmentationModalOpen] = useState(false);
    const [isAugmenting, setIsAugmenting] = useState(false);
    const [augmentationError, setAugmentationError] = useState<string | null>(null);
    const [augmentationSummary, setAugmentationSummary] = useState<string | null>(null);
    const [augmentationProgress, setAugmentationProgress] = useState<ClassificationAugmentationProgress>({
        status: 'idle',
        processed: 0,
        total: 0,
        percent: 0,
        added: 0
    });
    const [augmentationConfig, setAugmentationConfig] = useState<ClassificationAugmentationConfig>({
        target_scope: 'all',
        copies_per_image: 1,
        output_size: 640,
        apply_percent: 100,
        horizontal_flip_percent: 50,
        vertical_flip_percent: 0,
        hsv_h_percent: 1.5,
        hsv_s_percent: 40,
        hsv_v_percent: 40,
        erasing_percent: 0,
        auto_augment: '',
        force_color_jitter: false
    });
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
    const previewImageRef = useRef<HTMLImageElement | null>(null);
    const previewCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const detectionLayoutRef = useRef<HTMLDivElement | null>(null);
    const classificationGridRef = useRef<HTMLDivElement | null>(null);
    const detectionGridRef = useRef<HTMLDivElement | null>(null);
    const classificationSentinelRef = useRef<HTMLDivElement | null>(null);
    const detectionSentinelRef = useRef<HTMLDivElement | null>(null);
    const aspectUpdateBufferRef = useRef<Record<string, ImageOrientation>>({});
    const aspectUpdateRafRef = useRef<number | null>(null);
    const detectionActiveIndexRef = useRef(0);
    const augmentationRequestIdRef = useRef<string | null>(null);
    const pendingRef = useRef<PendingSave[]>([]);
    const flushRef = useRef(false);
    const undoRef = useRef<LabelChange[][]>([]);
    const classes = dataset?.classes ?? [];

    const classMap = useMemo(() => new Map(classes.map((cls) => [cls.id, cls.name])), [classes]);
    const classColorMap = useMemo(() => new Map(classes.map((cls) => [cls.id, cls.color || '#2fb8ff'])), [classes]);
    const duplicateGroups = validationReport?.duplicate_groups ?? [];
    const duplicateGroupsPreview = useMemo(() => duplicateGroups.slice(0, 20), [duplicateGroups]);
    const hiddenDuplicateGroupsCount = Math.max(duplicateGroups.length - duplicateGroupsPreview.length, 0);
    const validationProgress = useMemo(() => {
        const raw = Number(scanProgress?.progress ?? 0);
        if (!Number.isFinite(raw)) return 0;
        return Math.max(0, Math.min(100, raw));
    }, [scanProgress?.progress]);
    const validationProcessed = useMemo(() => {
        const raw = Number(scanProgress?.processed ?? scanProgress?.count ?? 0);
        if (!Number.isFinite(raw)) return 0;
        return Math.max(0, Math.trunc(raw));
    }, [scanProgress?.processed, scanProgress?.count]);
    const validationTotal = useMemo(() => {
        const raw = Number(scanProgress?.total ?? 0);
        if (!Number.isFinite(raw)) return 0;
        return Math.max(0, Math.trunc(raw));
    }, [scanProgress?.total]);
    const validationPhaseLabel = useMemo(() => {
        const phase = String(scanProgress?.phase || '').toLowerCase();
        const status = String(scanProgress?.status || '').toLowerCase();
        if (phase === 'scan' || status === 'scanning' || status === 'starting') return 'Scanning files';
        if (phase === 'validate' || status === 'running') return 'Validating images';
        if (status === 'completed') return 'Finalizing';
        return 'Processing dataset';
    }, [scanProgress?.phase, scanProgress?.status]);
    const validationMessage = useMemo(() => {
        const raw = String(scanProgress?.message || '').trim();
        if (raw) return raw;
        if (validationTotal > 0) return `Validated ${validationProcessed} / ${validationTotal} image(s).`;
        return 'Preparing validation pipeline...';
    }, [scanProgress?.message, validationProcessed, validationTotal]);
    const validationStaleSeconds = useMemo(() => {
        if (!validationLastUpdateAt) return 0;
        return Math.max(0, Math.floor((Date.now() - validationLastUpdateAt) / 1000));
    }, [validationLastUpdateAt, validationElapsedSec]);
    const saveStatusLabel = useMemo(() => {
        if (saveStatus === 'saving') return 'Saving...';
        if (saveStatus === 'pending') return pendingCount > 0 ? `Pending (${pendingCount})` : 'Pending';
        if (saveStatus === 'saved') return 'All changes saved';
        return 'Auto-save ready';
    }, [saveStatus, pendingCount]);

    const applyValidationReport = (report: ValidationReport) => {
        setValidationReport(report);
        setIsValidating(false);
        setScanProgress({
            phase: 'validate',
            status: 'completed',
            progress: 100,
            processed: report.valid_count + report.corrupted_count,
            total: report.valid_count + report.corrupted_count,
            message: 'Validation completed.'
        });
        setValidationLastUpdateAt(Date.now());
        const hasDuplicates = (report?.duplicate_groups_count || 0) > 0;
        if (hasDuplicates) {
            setDuplicateResolutionError(null);
            setIsDuplicateModalOpen(true);
        }
    };
    const clampDetectionSidebarWidth = useCallback((value: number) => {
        const minSidebar = 280;
        const fallbackMax = 560;
        const containerWidth = detectionLayoutRef.current?.getBoundingClientRect().width;
        if (!containerWidth) {
            return Math.max(minSidebar, Math.min(fallbackMax, value));
        }
        const maxSidebar = Math.max(minSidebar, containerWidth - 520);
        return Math.max(minSidebar, Math.min(maxSidebar, value));
    }, []);

    const applyPagination = (offset: number, total: number) => {
        setImageOffset(offset);
        setImageTotal(total);
    };
    const classificationThumbPixelSize = useMemo(() => {
        if (thumbSize === 'sm') return 192;
        if (thumbSize === 'xl') return 384;
        return 256;
    }, [thumbSize]);
    const getImageSrc = (image: DatasetImage) => {
        if (projectId) {
            return `${API_BASE_URL}/projects/${projectId}/dataset/images/${image.id}/file`;
        }
        return toFileUrl(image.path);
    };
    const getImageThumbnailSrc = (image: DatasetImage, size: number) => {
        if (projectId) {
            return `${API_BASE_URL}/projects/${projectId}/dataset/images/${image.id}/thumbnail?size=${Math.max(64, Math.trunc(size || 256))}`;
        }
        return toFileUrl(image.path);
    };

    const selectedImage = useMemo(
        () => images.find((image) => image.id === activeImageId) || null,
        [images, activeImageId]
    );
    const selectedImageId = selectedImage?.id ?? null;
    const selectedImageSrc = useMemo(() => {
        if (!selectedImage) return null;
        if (projectId) {
            return `${API_BASE_URL}/projects/${projectId}/dataset/images/${selectedImage.id}/file`;
        }
        return toFileUrl(selectedImage.path);
    }, [selectedImage, projectId]);
    const selectedImageMetadata = useMemo(
        () => (selectedImage ? imageMetadataById[selectedImage.id] : undefined),
        [selectedImage, imageMetadataById]
    );
    const previewImage = useMemo(
        () => images.find((image) => image.id === previewImageId) || null,
        [images, previewImageId]
    );
    const detectionPreviewImage = useMemo(() => {
        if (dataset?.task !== 'detection' || !previewImageId) return null;
        if (selectedImage && selectedImage.id === previewImageId) return selectedImage;
        return images.find((image) => image.id === previewImageId) || null;
    }, [dataset?.task, images, previewImageId, selectedImage]);
    const pageLabeledImages = useMemo(
        () => images.filter((image) => image.labels.length > 0),
        [images]
    );
    const pageUnlabeledImages = useMemo(
        () => images.filter((image) => image.labels.length === 0),
        [images]
    );
    const isVisibleByAugmentedFilter = useCallback((image: DatasetImage) => {
        if (classificationAugmentedFilter === 'all') return true;
        if (classificationAugmentedFilter === 'augmented') return Boolean(image.is_augmented);
        return !image.is_augmented;
    }, [classificationAugmentedFilter]);
    const filteredUnlabeledImages = useMemo(
        () => pageUnlabeledImages.filter(isVisibleByAugmentedFilter),
        [pageUnlabeledImages, isVisibleByAugmentedFilter]
    );
    const unlabeledSuggestionBreakdown = useMemo(() => {
        const counts = new Map<number, number>();
        const names = new Map<number, string>();
        const colors = new Map<number, string>();
        filteredUnlabeledImages.forEach((image) => {
            const suggestion = autoLabelSuggestionsByImage[image.id];
            if (!suggestion) return;
            const classId = Number(suggestion.predicted_class_id);
            if (!Number.isFinite(classId)) return;
            counts.set(classId, (counts.get(classId) || 0) + 1);
            const classMeta = classes.find((cls) => cls.id === classId);
            names.set(classId, classMeta?.name || suggestion.predicted_class_name || `Class ${classId}`);
            colors.set(classId, classMeta?.color || '#2fb8ff');
        });
        return Array.from(counts.entries())
            .map(([classId, count]) => ({
                classId,
                count,
                name: names.get(classId) || `Class ${classId}`,
                color: colors.get(classId) || '#2fb8ff'
            }))
            .sort((a, b) => {
                const diff = b.count - a.count;
                if (diff !== 0) return diff;
                return a.name.localeCompare(b.name);
            });
    }, [autoLabelSuggestionsByImage, classes, filteredUnlabeledImages]);
    const filteredUnlabeledImagesBySuggestion = useMemo(() => {
        if (unlabeledSuggestionFilter === null) return filteredUnlabeledImages;
        return filteredUnlabeledImages.filter(
            (image) => autoLabelSuggestionsByImage[image.id]?.predicted_class_id === unlabeledSuggestionFilter
        );
    }, [autoLabelSuggestionsByImage, filteredUnlabeledImages, unlabeledSuggestionFilter]);
    const unlabeledSuggestionTotal = useMemo(
        () => unlabeledSuggestionBreakdown.reduce((sum, item) => sum + item.count, 0),
        [unlabeledSuggestionBreakdown]
    );
    const labeledImagesByAugmentedFilter = useMemo(
        () => pageLabeledImages.filter(isVisibleByAugmentedFilter),
        [pageLabeledImages, isVisibleByAugmentedFilter]
    );
    const filteredLabeledImages = useMemo(() => {
        if (labeledClassFilter === null) return labeledImagesByAugmentedFilter;
        return labeledImagesByAugmentedFilter.filter((image) => image.labels.includes(labeledClassFilter));
    }, [labeledImagesByAugmentedFilter, labeledClassFilter]);
    const visibleImages = classificationTab === 'labeled' ? filteredLabeledImages : filteredUnlabeledImagesBySuggestion;
    const pendingSuggestionIds = useMemo(
        () => Object.keys(autoLabelSuggestionsByImage),
        [autoLabelSuggestionsByImage]
    );
    const pendingSuggestionCount = pendingSuggestionIds.length;
    const selectedSuggestionIds = useMemo(
        () => Array.from(selectedIds).filter((id) => Boolean(autoLabelSuggestionsByImage[id])),
        [selectedIds, autoLabelSuggestionsByImage]
    );
    const selectedSuggestionCount = selectedSuggestionIds.length;
    const autoLabelModelOptions = useMemo(
        () => autoLabelModels.map((model) => ({ value: model.id, label: model.name || model.id })),
        [autoLabelModels]
    );
    const renderedClassificationImages = useMemo(
        () => visibleImages.slice(0, classificationRenderCount),
        [visibleImages, classificationRenderCount]
    );
    const canLoadMoreClassification = renderedClassificationImages.length < visibleImages.length;
    const previewImageIndex = useMemo(
        () => visibleImages.findIndex((image) => image.id === previewImageId),
        [visibleImages, previewImageId]
    );
    const canGoPrevPreviewImage = previewImageIndex > 0;
    const canGoNextPreviewImage =
        previewImageIndex >= 0 && previewImageIndex < visibleImages.length - 1;
    const classificationAugmentedCount = useMemo(
        () => images.filter((image) => Boolean(image.is_augmented)).length,
        [images]
    );
    const classificationOriginalCount = Math.max(images.length - classificationAugmentedCount, 0);
    const matchesAugmentationScope = useCallback((image: DatasetImage) => {
        if (augmentationConfig.target_scope === 'labeled') return image.labels.length > 0;
        if (augmentationConfig.target_scope === 'unlabeled') return image.labels.length === 0;
        return true;
    }, [augmentationConfig.target_scope]);
    const augmentationScopeCount = useMemo(
        () => images.filter((image) => matchesAugmentationScope(image)).length,
        [images, matchesAugmentationScope]
    );
    const augmentationSelectedScopeCount = useMemo(
        () => images.filter((image) => selectedIds.has(image.id) && matchesAugmentationScope(image)).length,
        [images, selectedIds, matchesAugmentationScope]
    );
    const detectionLabeledImages = useMemo(
        () => images.filter((image) => Boolean(image.detection_done)),
        [images]
    );
    const detectionUnlabeledImages = useMemo(
        () => images.filter((image) => !image.detection_done),
        [images]
    );
    const filteredDetectionLabeledImages = useMemo(() => {
        if (detectionLabeledClassFilters.length === 0) return detectionLabeledImages;
        const classFilterSet = new Set(detectionLabeledClassFilters);
        return detectionLabeledImages.filter((image) =>
            image.boxes.some((box) => classFilterSet.has(box.class_id))
        );
    }, [detectionLabeledImages, detectionLabeledClassFilters]);
    const detectionVisibleImages =
        detectionTab === 'labeled' ? filteredDetectionLabeledImages : detectionUnlabeledImages;
    const renderedDetectionImages = useMemo(
        () => detectionVisibleImages.slice(0, detectionRenderCount),
        [detectionVisibleImages, detectionRenderCount]
    );
    const canLoadMoreDetection = renderedDetectionImages.length < detectionVisibleImages.length;
    const detectionTotal = images.length;
    const detectionDone = detectionLabeledImages.length;
    const detectionRemaining = Math.max(detectionTotal - detectionDone, 0);
    const detectionProgressPercent = detectionTotal > 0
        ? Math.min(100, Math.round((detectionDone / detectionTotal) * 100))
        : 0;
    const detectionRemainingPercent = detectionTotal > 0
        ? Math.min(100, Math.round((detectionRemaining / detectionTotal) * 100))
        : 0;
    const { detectionBalanceTotal, detectionBalanceBreakdown } = useMemo(() => {
        const counts = new Map<number, number>();
        classes.forEach((cls) => counts.set(cls.id, 0));
        images.forEach((image) => {
            image.boxes.forEach((box) => {
                counts.set(box.class_id, (counts.get(box.class_id) || 0) + 1);
            });
        });
        const totalBoxes = Array.from(counts.values()).reduce((sum, value) => sum + value, 0);
        const maxCount = Math.max(0, ...Array.from(counts.values()));
        const breakdown = classes.map((cls) => {
            const count = counts.get(cls.id) || 0;
            const portion = totalBoxes > 0 ? (count / totalBoxes) * 100 : 0;
            const relative = maxCount > 0 ? (count / maxCount) * 100 : 0;
            return {
                id: cls.id,
                name: cls.name,
                color: cls.color || '#2fb8ff',
                count,
                portion,
                relative
            };
        });
        return { detectionBalanceTotal: totalBoxes, detectionBalanceBreakdown: breakdown };
    }, [images, classes]);
    const detectionLabeledClassImageCount = useMemo(() => {
        const counts = new Map<number, number>();
        classes.forEach((cls) => counts.set(cls.id, 0));
        detectionLabeledImages.forEach((image) => {
            const classesInImage = new Set(image.boxes.map((box) => box.class_id));
            classesInImage.forEach((classId) => {
                counts.set(classId, (counts.get(classId) || 0) + 1);
            });
        });
        return counts;
    }, [classes, detectionLabeledImages]);
    const activeDetectionIndex = useMemo(
        () => detectionVisibleImages.findIndex((image) => image.id === activeImageId),
        [detectionVisibleImages, activeImageId]
    );
    const canGoPrevDetection = activeDetectionIndex > 0;
    const canGoNextDetection =
        activeDetectionIndex >= 0 && activeDetectionIndex < detectionVisibleImages.length - 1;
    const detectionPreviewIndex = useMemo(
        () => detectionVisibleImages.findIndex((image) => image.id === previewImageId),
        [detectionVisibleImages, previewImageId]
    );
    const canGoPrevDetectionPreview = detectionPreviewIndex > 0;
    const canGoNextDetectionPreview =
        detectionPreviewIndex >= 0 && detectionPreviewIndex < detectionVisibleImages.length - 1;
    const isDetectionPreviewOpen = Boolean(dataset?.task === 'detection' && previewImageId);
    const progressImages = useMemo(() => {
        if (!dataset?.images || dataset.images.length === 0) return images;
        if (dataset.images.length === images.length) return images;
        if (images.length === 0) return dataset.images;
        const override = new Map(images.map((image) => [image.id, image]));
        return dataset.images.map((image) => override.get(image.id) ?? image);
    }, [dataset?.images, images]);
    const {
        progressTotal,
        progressLabeled,
        progressRemaining,
        progressPercent,
        balanceTotal,
        balanceBreakdown,
        balanceById
    } = useMemo(() => {
        const total = progressImages.length;
        let labeled = 0;
        const counts = new Map<number, number>();
        classes.forEach((cls) => counts.set(cls.id, 0));
        progressImages.forEach((image) => {
            if (image.labels.length > 0) labeled += 1;
            image.labels.forEach((id) => counts.set(id, (counts.get(id) || 0) + 1));
        });
        const totalAssignments = Array.from(counts.values()).reduce((sum, value) => sum + value, 0);
        const percent = total > 0 ? Math.min(100, Math.round((labeled / total) * 100)) : 0;
        const maxCount = Math.max(0, ...Array.from(counts.values()));
        const breakdown = classes.map((cls) => {
            const count = counts.get(cls.id) || 0;
            const portion = totalAssignments > 0 ? (count / totalAssignments) * 100 : 0;
            const relative = maxCount > 0 ? (count / maxCount) * 100 : 0;
            return {
                id: cls.id,
                name: cls.name,
                color: cls.color || '#2fb8ff',
                count,
                portion,
                relative
            };
        });
        return {
            progressTotal: total,
            progressLabeled: labeled,
            progressRemaining: Math.max(total - labeled, 0),
            progressPercent: percent,
            balanceTotal: totalAssignments,
            balanceBreakdown: breakdown,
            balanceById: counts
        };
    }, [progressImages, classes]);

    useEffect(() => {
        setClassificationRenderCount(CLASSIFICATION_INITIAL_RENDER);
    }, [classificationTab, labeledClassFilter, unlabeledSuggestionFilter, classificationAugmentedFilter, imageOffset, images.length]);

    useEffect(() => {
        setDetectionRenderCount(DETECTION_INITIAL_RENDER);
    }, [detectionTab, detectionLabeledClassFilters, imageOffset, images.length]);

    useEffect(() => {
        const root = classificationGridRef.current;
        const sentinel = classificationSentinelRef.current;
        if (!root || !sentinel) return;
        if (!canLoadMoreClassification) return;

        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (!entry.isIntersecting) return;
                    setClassificationRenderCount((prev) => Math.min(visibleImages.length, prev + CLASSIFICATION_RENDER_CHUNK));
                });
            },
            { root, rootMargin: '320px 0px', threshold: 0.01 }
        );

        observer.observe(sentinel);
        return () => observer.disconnect();
    }, [canLoadMoreClassification, visibleImages.length]);

    useEffect(() => {
        const root = detectionGridRef.current;
        const sentinel = detectionSentinelRef.current;
        if (!root || !sentinel) return;
        if (!canLoadMoreDetection) return;

        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (!entry.isIntersecting) return;
                    setDetectionRenderCount((prev) => Math.min(detectionVisibleImages.length, prev + DETECTION_RENDER_CHUNK));
                });
            },
            { root, rootMargin: '260px 0px', threshold: 0.01 }
        );

        observer.observe(sentinel);
        return () => observer.disconnect();
    }, [canLoadMoreDetection, detectionVisibleImages.length]);

    useEffect(() => () => {
        if (aspectUpdateRafRef.current !== null) {
            window.cancelAnimationFrame(aspectUpdateRafRef.current);
            aspectUpdateRafRef.current = null;
        }
    }, []);

    const handleThumbLoad = (imageId: string, event: React.SyntheticEvent<HTMLImageElement>) => {
        const target = event.currentTarget;
        if (!target.naturalWidth || !target.naturalHeight) return;
        const ratio = target.naturalWidth / target.naturalHeight;
        const nextAspect: 'portrait' | 'landscape' | 'square' =
            ratio > 1.15 ? 'landscape' : ratio < 0.85 ? 'portrait' : 'square';
        aspectUpdateBufferRef.current[imageId] = nextAspect;
        if (aspectUpdateRafRef.current !== null) return;
        aspectUpdateRafRef.current = window.requestAnimationFrame(() => {
            const pending = aspectUpdateBufferRef.current;
            aspectUpdateBufferRef.current = {};
            aspectUpdateRafRef.current = null;
            const entries = Object.entries(pending);
            if (entries.length === 0) return;
            setImageAspect((prev) => {
                let changed = false;
                const next = { ...prev };
                entries.forEach(([id, aspect]) => {
                    if (next[id] === aspect) return;
                    next[id] = aspect;
                    changed = true;
                });
                return changed ? next : prev;
            });
        });
    };

    const handleSelectedImageLoad = (event: React.SyntheticEvent<HTMLImageElement>) => {
        if (!selectedImage) return;
        const target = event.currentTarget;
        const width = target.naturalWidth;
        const height = target.naturalHeight;
        if (!width || !height) return;
        const orientation = getOrientationFromSize(width, height);
        const megapixels = (width * height) / 1_000_000;
        setImageMetadataById((prev) => {
            const current = prev[selectedImage.id];
            if (
                current?.width === width &&
                current?.height === height &&
                current?.orientation === orientation
            ) {
                return prev;
            }
            return {
                ...prev,
                [selectedImage.id]: {
                    ...current,
                    width,
                    height,
                    orientation,
                    megapixels
                }
            };
        });
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
                    (item.type === 'labels' || item.type === 'boxes' || item.type === 'detection_done') &&
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
            } else if (item.type === 'boxes') {
                byId.set(item.imageId, { ...image, boxes: item.boxes });
            } else {
                byId.set(item.imageId, { ...image, detection_done: item.detection_done });
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
                } else if (item.type === 'boxes') {
                    res = await fetch(
                        `${API_BASE_URL}/projects/${projectId}/dataset/images/${item.imageId}/boxes`,
                        {
                            method: 'PUT',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ boxes: item.boxes })
                        }
                    );
                } else {
                    res = await fetch(
                        `${API_BASE_URL}/projects/${projectId}/dataset/images/${item.imageId}/status`,
                        {
                            method: 'PUT',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ detection_done: item.detection_done })
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

    const pushUndo = (changes: LabelChange[]) => {
        if (changes.length === 0) return;
        const next = undoRef.current.slice();
        next.push(changes);
        if (next.length > 5) {
            next.shift();
        }
        undoRef.current = next;
    };

    const applyUndo = (changes: LabelChange[]) => {
        if (!projectId || changes.length === 0) return;
        const changeMap = new Map(changes.map((change) => [change.imageId, change]));
        setImages((prev) =>
            prev.map((image) => {
                const change = changeMap.get(image.id);
                if (!change) return image;
                return { ...image, labels: change.prevLabels };
            })
        );
        const updatedAt = Date.now();
        enqueuePendingBatch(
            changes.map((change) => ({
                type: 'labels',
                imageId: change.imageId,
                labels: change.prevLabels,
                updatedAt
            }))
        );
    };

    const loadImages = async (offset: number, total?: number) => {
        if (!projectId) return;
        const imagesRes = await fetch(
            `${API_BASE_URL}/projects/${projectId}/dataset/images?limit=${IMAGE_PAGE_SIZE}&offset=${offset}`
        );
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
            if (typeof total === 'number') {
                applyPagination(offset, total);
            } else {
                setImageOffset(offset);
            }
        }
    };

    const fetchDataset = async () => {
        if (!projectId) return;
        setLoading(true);
        setDatasetError(null);
        try {
            const res = await fetch(`${API_BASE_URL}/projects/${projectId}/dataset`);
            if (!res.ok) {
                throw new Error(await readErrorMessage(res, 'Failed to load project dataset.'));
            }
            const data = await res.json();
            setDataset(data);
            const total = Array.isArray(data.images) ? data.images.length : 0;
            applyPagination(0, total);
            if (pendingRef.current.length === 0) {
                const pending = readPendingSaves(projectId);
                pendingRef.current = pending;
                setPendingCount(pending.length);
                if (pending.length > 0) {
                    setSaveStatus('pending');
                }
            }
            const baseImages = applyPendingToImages(data.images ?? [], pendingRef.current);
            setImages(baseImages);
            if (pendingRef.current.length > 0) {
                void flushPending();
            } else {
                setSaveStatus('saved');
            }
        } catch (error: any) {
            setDataset(null);
            setImages([]);
            setDatasetError(error?.message || 'Failed to load project dataset.');
            setSaveStatus('idle');
        } finally {
            setLoading(false);
        }
    };

    const fetchAutoLabelModels = useCallback(async () => {
        if (!projectId) return;
        setAutoLabelLoadingModels(true);
        setAutoLabelError(null);
        try {
            const res = await fetch(`${API_BASE_URL}/projects/${projectId}/dataset/autolabel/classification/models`);
            if (!res.ok) {
                const text = await res.text();
                throw new Error(text || 'Failed to load classification models');
            }
            const payload = await res.json();
            const models: AutoLabelModelOption[] = Array.isArray(payload?.models) ? payload.models : [];
            setAutoLabelModels(models);
            setAutoLabelModelId((current) => {
                if (current && models.some((model) => model.id === current)) return current;
                return models[0]?.id || '';
            });
            if (models.length === 0) {
                setAutoLabelSuggestionsByImage({});
            }
        } catch (error: any) {
            setAutoLabelError(error?.message || 'Failed to load classification models');
            setAutoLabelModels([]);
            setAutoLabelModelId('');
            setAutoLabelSuggestionsByImage({});
        } finally {
            setAutoLabelLoadingModels(false);
        }
    }, [projectId]);

    const fetchAutoLabelSuggestions = useCallback(async (variantId?: string) => {
        if (!projectId) return;
        const resolvedVariantId = (variantId || autoLabelModelId || '').trim();
        if (!resolvedVariantId) {
            setAutoLabelSuggestionsByImage({});
            return;
        }
        try {
            const qs = new URLSearchParams({
                variant_id: resolvedVariantId,
                status: 'pending',
                only_unlabeled: 'true'
            });
            const res = await fetch(
                `${API_BASE_URL}/projects/${projectId}/dataset/autolabel/classification/suggestions?${qs.toString()}`
            );
            if (!res.ok) {
                const text = await res.text();
                throw new Error(text || 'Failed to load auto-label suggestions');
            }
            const payload = await res.json();
            const suggestions = payload?.suggestions && typeof payload.suggestions === 'object'
                ? payload.suggestions as Record<string, AutoLabelSuggestion>
                : {};
            setAutoLabelSuggestionsByImage(suggestions);
        } catch (error: any) {
            setAutoLabelError(error?.message || 'Failed to load auto-label suggestions');
            setAutoLabelSuggestionsByImage({});
        }
    }, [autoLabelModelId, projectId]);

    const runAutoLabelClassification = useCallback(async () => {
        if (!projectId || !autoLabelModelId) return;
        const selectedUnlabeledIds = Array.from(selectedIds).filter((imageId) => {
            const image = images.find((item) => item.id === imageId);
            return Boolean(image) && (image?.labels?.length || 0) === 0;
        });
        const targetIds = selectedUnlabeledIds.length > 0
            ? selectedUnlabeledIds
            : images.filter((image) => (image.labels?.length || 0) === 0).map((image) => image.id);
        if (targetIds.length === 0) {
            setAutoLabelNotice('No unlabeled images available for auto-classification.');
            return;
        }
        setAutoLabelRunning(true);
        setAutoLabelError(null);
        setAutoLabelNotice(null);
        setAutoLabelProgress({
            status: 'starting',
            processed: 0,
            total: targetIds.length,
            progress: 0,
            message: 'Starting auto-classification...'
        });
        try {
            const res = await fetch(`${API_BASE_URL}/projects/${projectId}/dataset/autolabel/classification/run`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    variant_id: autoLabelModelId,
                    image_ids: targetIds,
                    only_unlabeled: true,
                    min_confidence: Math.max(0, Math.min(1, autoLabelMinConfidence))
                })
            });
            if (!res.ok) {
                const text = await res.text();
                throw new Error(text || 'Auto-classification failed');
            }
            const payload = await res.json();
            setAutoLabelNotice(
                `Auto-classification completed: ${payload?.suggested ?? 0} suggestion(s), ${payload?.errors ?? 0} error(s).`
            );
            await fetchAutoLabelSuggestions(autoLabelModelId);
        } catch (error: any) {
            setAutoLabelError(error?.message || 'Auto-classification failed');
            setAutoLabelProgress((prev) => ({
                ...prev,
                status: 'error',
                message: error?.message || 'Auto-classification failed'
            }));
        } finally {
            setAutoLabelRunning(false);
        }
    }, [autoLabelMinConfidence, autoLabelModelId, fetchAutoLabelSuggestions, images, projectId, selectedIds]);

    const handleApplyAutoLabelAction = useCallback(async (action: 'accept' | 'reject' | 'clear') => {
        if (!projectId || !autoLabelModelId) return;
        if (selectedSuggestionIds.length === 0) {
            setAutoLabelNotice('Select images with suggestions before applying this action.');
            return;
        }
        setAutoLabelError(null);
        try {
            const res = await fetch(`${API_BASE_URL}/projects/${projectId}/dataset/autolabel/classification/apply`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    variant_id: autoLabelModelId,
                    image_ids: selectedSuggestionIds,
                    action
                })
            });
            if (!res.ok) {
                const text = await res.text();
                throw new Error(text || 'Failed to apply auto-label action');
            }
            const payload = await res.json();
            if (action === 'accept') {
                await fetchDataset();
            }
            await fetchAutoLabelSuggestions(autoLabelModelId);
            setSelectedIds((prev) => {
                const next = new Set(prev);
                selectedSuggestionIds.forEach((id) => next.delete(id));
                return next;
            });
            const label = action === 'accept' ? 'accepted' : action === 'reject' ? 'rejected' : 'cleared';
            setAutoLabelNotice(`${payload?.changed ?? 0} suggestion(s) ${label}.`);
        } catch (error: any) {
            setAutoLabelError(error?.message || 'Failed to apply auto-label action');
        }
    }, [autoLabelModelId, fetchAutoLabelSuggestions, fetchDataset, projectId, selectedSuggestionIds]);

    const handleAcceptAutoLabelSuggestion = useCallback(async (imageId: string, variantId?: string) => {
        if (!projectId) return false;
        const resolvedVariantId = String(variantId || autoLabelModelId || '').trim();
        if (!resolvedVariantId) return false;
        setAutoLabelError(null);
        try {
            const res = await fetch(`${API_BASE_URL}/projects/${projectId}/dataset/autolabel/classification/apply`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    variant_id: resolvedVariantId,
                    image_ids: [imageId],
                    action: 'accept'
                })
            });
            if (!res.ok) {
                const text = await res.text();
                throw new Error(text || 'Failed to accept suggestion');
            }
            const payload = await res.json();
            await fetchDataset();
            await fetchAutoLabelSuggestions(resolvedVariantId);
            setSelectedIds((prev) => {
                const next = new Set(prev);
                next.delete(imageId);
                return next;
            });
            if (Number(payload?.changed || 0) > 0) {
                setAutoLabelNotice('Suggestion accepted and image moved to labeled.');
                return true;
            }
            return false;
        } catch (error: any) {
            setAutoLabelError(error?.message || 'Failed to accept suggestion');
            return false;
        }
    }, [autoLabelModelId, fetchAutoLabelSuggestions, fetchDataset, projectId]);

    const ensureClassId = async (name: string) => {
        if (!projectId || !dataset) return null;
        const normalized = name.trim();
        if (!normalized) return null;
        const existing = dataset.classes.find(
            (cls) => cls.name.toLowerCase() === normalized.toLowerCase()
        );
        if (existing) return existing.id;
        const res = await fetch(`${API_BASE_URL}/projects/${projectId}/dataset/classes`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ classes: [...dataset.classes, { name: normalized }] })
        });
        if (!res.ok) return null;
        const updated = await res.json();
        setDataset(updated);
        const created = updated.classes.find(
            (cls: DatasetClass) => cls.name.toLowerCase() === normalized.toLowerCase()
        );
        return created?.id ?? null;
    };

    const runPreclassify = async () => {
        if (!projectId) return;
        const targetIds = selectedIds.size > 0
            ? Array.from(selectedIds)
            : visibleImages.map((image) => image.id);
        if (targetIds.length === 0) return;
        setIsPreclassifying(true);
        setPreclassifyError(null);
        try {
            const res = await fetch(`${API_BASE_URL}/projects/${projectId}/dataset/preclassify`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_ids: targetIds,
                    model: preclassifyModel,
                    topk: preclassifyTopK
                })
            });
            if (!res.ok) {
                throw new Error('Pre-classification failed');
            }
            const data = await res.json();
            const results = data?.results || {};
            setPreclassifyResults((prev) => {
                const next = { ...prev };
                Object.entries(results).forEach(([imageId, payload]) => {
                    const predictions = (payload as any).predictions || [];
                    next[imageId] = predictions;
                });
                return next;
            });
        } catch (err: any) {
            setPreclassifyError(err?.message || 'Pre-classification failed');
        } finally {
            setIsPreclassifying(false);
        }
    };

    const handleAcceptPrelabel = async (imageId: string, prediction: PreclassifyPrediction) => {
        const classId = await ensureClassId(prediction.class_name);
        if (classId === null) return;
        await handleAssignClassSingle(imageId, classId);
    };

    const updateAugmentationConfig = <K extends keyof ClassificationAugmentationConfig>(
        key: K,
        value: ClassificationAugmentationConfig[K]
    ) => {
        setAugmentationConfig((prev) => ({ ...prev, [key]: value }));
    };

    const runClassificationAugmentation = async () => {
        if (!projectId || dataset?.task !== 'classification') return;
        const scopedImages = images.filter((image) => matchesAugmentationScope(image));
        const scopedIdSet = new Set(scopedImages.map((image) => image.id));
        const targetIds = selectedIds.size > 0
            ? Array.from(selectedIds).filter((id) => scopedIdSet.has(id))
            : scopedImages.map((image) => image.id);
        if (targetIds.length === 0) {
            setAugmentationError('No images match the selected augmentation scope.');
            return;
        }
        setIsAugmenting(true);
        setAugmentationError(null);
        setAugmentationSummary(null);
        const requestId = typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
            ? crypto.randomUUID()
            : `aug_${Date.now()}_${Math.random().toString(16).slice(2)}`;
        augmentationRequestIdRef.current = requestId;
        const plannedTotal = Math.max(1, targetIds.length * Math.max(1, Math.trunc(augmentationConfig.copies_per_image)));
        setAugmentationProgress({
            status: 'starting',
            processed: 0,
            total: plannedTotal,
            percent: 0,
            added: 0
        });
        try {
            const payload = {
                request_id: requestId,
                image_ids: targetIds,
                target_scope: augmentationConfig.target_scope,
                copies_per_image: Math.max(1, Math.trunc(augmentationConfig.copies_per_image)),
                output_size: Math.max(32, Math.trunc(augmentationConfig.output_size)),
                apply_percent: clampPercent(augmentationConfig.apply_percent),
                horizontal_flip_percent: clampPercent(augmentationConfig.horizontal_flip_percent),
                vertical_flip_percent: clampPercent(augmentationConfig.vertical_flip_percent),
                hsv_h_percent: clampPercent(augmentationConfig.hsv_h_percent),
                hsv_s_percent: clampPercent(augmentationConfig.hsv_s_percent),
                hsv_v_percent: clampPercent(augmentationConfig.hsv_v_percent),
                erasing_percent: clampPercent(augmentationConfig.erasing_percent),
                auto_augment: augmentationConfig.auto_augment || null,
                force_color_jitter: augmentationConfig.force_color_jitter
            };
            const res = await fetch(`${API_BASE_URL}/projects/${projectId}/dataset/augment/classification`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!res.ok) {
                const errorPayload = await res.json().catch(() => ({}));
                throw new Error(errorPayload?.detail || 'Classification augmentation failed');
            }
            const data = await res.json();
            await fetchDataset();
            setSelectedIds(new Set());
            setAugmentationSummary(
                `Generated ${data?.added ?? 0} augmented image(s) from ${data?.generated_from ?? targetIds.length} source image(s).`
            );
            setAugmentationProgress((prev) => ({
                ...prev,
                status: 'completed',
                processed: prev.total || plannedTotal,
                total: prev.total || plannedTotal,
                percent: 100,
                added: data?.added ?? prev.added
            }));
            augmentationRequestIdRef.current = null;
        } catch (err: any) {
            setAugmentationError(err?.message || 'Classification augmentation failed');
            setAugmentationProgress((prev) => ({
                ...prev,
                status: 'idle'
            }));
            augmentationRequestIdRef.current = null;
        } finally {
            setIsAugmenting(false);
        }
    };

    useEffect(() => {
        pendingRef.current = [];
        setPendingCount(0);
        setSaveStatus('idle');
        setSaveError(null);
        setDatasetError(null);
        setScanProgress(null);
        setValidationReport(null);
        setValidationElapsedSec(0);
        setValidationLastUpdateAt(null);
        setIsDuplicateModalOpen(false);
        setDuplicateResolutionError(null);
        setIsResolvingDuplicates(false);
        setImageOffset(0);
        setImageTotal(0);
        setPreclassifyResults({});
        setImageMetadataById({});
        setIsAugmentationModalOpen(false);
        setIsAugmenting(false);
        setAugmentationError(null);
        setAugmentationSummary(null);
        setAutoLabelModels([]);
        setAutoLabelModelId('');
        setAutoLabelSuggestionsByImage({});
        setAutoLabelError(null);
        setAutoLabelNotice(null);
        setAutoLabelRunning(false);
        setAutoLabelProgress({
            status: 'idle',
            processed: 0,
            total: 0,
            progress: 0,
            message: ''
        });
        setAugmentationProgress({
            status: 'idle',
            processed: 0,
            total: 0,
            percent: 0,
            added: 0
        });
        augmentationRequestIdRef.current = null;
        setClassificationAugmentedFilter('all');
    }, [projectId]);

    useEffect(() => {
        fetchDataset();
    }, [projectId]);

    useEffect(() => {
        if (dataset?.task !== 'classification') {
            setAutoLabelModels([]);
            setAutoLabelModelId('');
            setAutoLabelSuggestionsByImage({});
            return;
        }
        void fetchAutoLabelModels();
    }, [dataset?.task, fetchAutoLabelModels]);

    useEffect(() => {
        if (dataset?.task !== 'classification') return;
        if (!autoLabelModelId) {
            setAutoLabelSuggestionsByImage({});
            return;
        }
        void fetchAutoLabelSuggestions(autoLabelModelId);
    }, [autoLabelModelId, dataset?.task, fetchAutoLabelSuggestions]);

    useEffect(() => {
        if (dataset?.task !== 'classification') return;
        const hasActive = activeImageId ? images.some((image) => image.id === activeImageId) : false;
        if (!hasActive && images.length > 0) {
            setActiveImageId(images[0].id);
        }
    }, [dataset?.task, images, activeImageId]);

    useEffect(() => {
        if (dataset?.task !== 'detection') return;
        if (classes.length === 0) {
            setActiveClassId(null);
            return;
        }
        if (activeClassId === null || !classes.some((cls) => cls.id === activeClassId)) {
            setActiveClassId(classes[0].id);
        }
    }, [dataset?.task, classes, activeClassId]);

    useEffect(() => {
        if (dataset?.task !== 'detection' || !activeImageId) return;
        const activeIndex = detectionVisibleImages.findIndex((image) => image.id === activeImageId);
        if (activeIndex >= 0) {
            detectionActiveIndexRef.current = activeIndex;
        }
    }, [dataset?.task, detectionVisibleImages, activeImageId]);

    useEffect(() => {
        if (dataset?.task !== 'detection') return;
        if (previewImageId) {
            if (activeImageId !== previewImageId) {
                setActiveImageId(previewImageId);
            }
            return;
        }
        if (detectionVisibleImages.length === 0) {
            setActiveImageId(null);
            detectionActiveIndexRef.current = 0;
            return;
        }
        const hasActive = activeImageId
            ? detectionVisibleImages.some((image) => image.id === activeImageId)
            : false;
        if (!hasActive) {
            const fallbackIndex = Math.min(
                Math.max(detectionActiveIndexRef.current, 0),
                detectionVisibleImages.length - 1
            );
            setActiveImageId(detectionVisibleImages[fallbackIndex].id);
        }
    }, [dataset?.task, detectionVisibleImages, activeImageId, previewImageId]);

    useEffect(() => {
        setDetectionLabeledClassFilters((prev) =>
            prev.filter((classId) => classes.some((cls) => cls.id === classId))
        );
    }, [classes]);

    useEffect(() => {
        const handleWindowResize = () => {
            setDetectionSidebarWidth((current) => clampDetectionSidebarWidth(current));
        };
        window.addEventListener('resize', handleWindowResize);
        return () => window.removeEventListener('resize', handleWindowResize);
    }, [clampDetectionSidebarWidth]);

    useEffect(() => {
        if (!isResizingDetectionSidebar) return;
        const handleMouseMove = (event: MouseEvent) => {
            const containerRect = detectionLayoutRef.current?.getBoundingClientRect();
            if (!containerRect) return;
            const nextWidth = event.clientX - containerRect.left;
            setDetectionSidebarWidth(clampDetectionSidebarWidth(nextWidth));
        };
        const handleMouseUp = () => setIsResizingDetectionSidebar(false);
        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('mouseup', handleMouseUp);
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        };
    }, [isResizingDetectionSidebar, clampDetectionSidebarWidth]);

    useEffect(() => {
        setEditingBoxIndex(null);
        setEditingBoxDraft(null);
        setResizeDrag(null);
    }, [activeImageId]);

    useEffect(() => {
        if (!selectedImageId || !selectedImageSrc) return;
        if (typeof selectedImageMetadata?.fileSizeBytes !== 'undefined') return;
        let cancelled = false;
        const loadFileSize = async () => {
            let fileSizeBytes: number | null = null;
            try {
                const response = await fetch(selectedImageSrc, { method: 'HEAD' });
                if (response.ok) {
                    const value = response.headers.get('content-length');
                    const parsed = value ? Number(value) : NaN;
                    if (Number.isFinite(parsed) && parsed > 0) {
                        fileSizeBytes = parsed;
                    }
                }
            } catch {
                fileSizeBytes = null;
            }
            if (cancelled) return;
            setImageMetadataById((prev) => {
                const previous = prev[selectedImageId] ?? {};
                if (typeof previous.fileSizeBytes !== 'undefined') {
                    return prev;
                }
                return {
                    ...prev,
                    [selectedImageId]: {
                        ...previous,
                        fileSizeBytes
                    }
                };
            });
        };
        void loadFileSize();
        return () => {
            cancelled = true;
        };
    }, [selectedImageId, selectedImageSrc, selectedImageMetadata?.fileSizeBytes]);

    useEffect(() => {
        const handleMessage = (event: MessageEvent) => {
            try {
                const msg = JSON.parse(event.data);
                if (msg.event === 'dataset/scan_progress') {
                    setScanProgress(msg.data);
                    setValidationLastUpdateAt(Date.now());
                } else if (msg.event === 'dataset/scan_completed') {
                    setScanProgress((prev: any) => ({
                        ...(prev || {}),
                        phase: 'scan',
                        status: 'completed',
                        progress: 100,
                        count: Number(msg.data?.total_images || prev?.count || 0),
                        message: `Scan completed with ${Number(msg.data?.total_images || 0)} image(s). Starting validation...`
                    }));
                    setValidationLastUpdateAt(Date.now());
                } else if (msg.event === 'dataset/validate_progress') {
                    setScanProgress(msg.data);
                    setValidationLastUpdateAt(Date.now());
                } else if (msg.event === 'dataset/validated') {
                    applyValidationReport(msg.data as ValidationReport);
                } else if (msg.event === 'dataset/augment_progress') {
                    const data = msg.data || {};
                    if (projectId && data.project_id && data.project_id !== projectId) return;
                    const activeRequestId = augmentationRequestIdRef.current;
                    if (!activeRequestId) return;
                    if (data.request_id !== activeRequestId) return;
                    const nextStatus: ClassificationAugmentationProgress['status'] =
                        data.status === 'completed'
                            ? 'completed'
                            : data.status === 'starting'
                                ? 'starting'
                                : data.status === 'running'
                                    ? 'running'
                                    : 'running';
                    setAugmentationProgress({
                        status: nextStatus,
                        processed: Number(data.processed || 0),
                        total: Number(data.total || 0),
                        percent: Number(data.percent || 0),
                        added: Number(data.added || 0)
                    });
                    if (nextStatus === 'completed') {
                        augmentationRequestIdRef.current = null;
                    }
                } else if (msg.event === 'dataset/autolabel_progress') {
                    const data = msg.data || {};
                    if (projectId && data.project_id && data.project_id !== projectId) return;
                    const nextStatus = String(data.status || 'running').toLowerCase();
                    setAutoLabelProgress({
                        status: nextStatus === 'completed'
                            ? 'completed'
                            : nextStatus === 'starting'
                                ? 'starting'
                                : nextStatus === 'error'
                                    ? 'error'
                                    : 'running',
                        processed: Number(data.processed || 0),
                        total: Number(data.total || 0),
                        progress: Number(data.progress || 0),
                        message: String(data.message || ''),
                        suggested: Number(data.suggested || 0),
                        removed: Number(data.removed || 0),
                        errors: Number(data.errors || 0)
                    });
                    if (nextStatus === 'completed' || nextStatus === 'error') {
                        setAutoLabelRunning(false);
                    }
                }
            } catch {
                // ignore
            }
        };

        const ws = new WebSocket(ENGINE_WS_URL);
        ws.onmessage = handleMessage;

        return () => ws.close();
    }, [projectId]);

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
        if (!window.ipcRenderer?.openDirectory) {
            setDatasetError('Desktop file dialog is unavailable. Restart the app and try again.');
            return;
        }
        const path = await window.ipcRenderer?.openDirectory?.();
        if (!path) return;
        setIngesting(true);
        setDatasetError(null);
        try {
            const response = await fetch(`${API_BASE_URL}/projects/${projectId}/dataset/ingest/directories`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ paths: [path] })
            });
            if (!response.ok) {
                throw new Error(await readErrorMessage(response, 'Failed to import directory.'));
            }
            await fetchDataset();
        } catch (error: any) {
            setDatasetError(error?.message || 'Failed to import directory.');
        } finally {
            setIngesting(false);
        }
    };

    const handleIngestFiles = async () => {
        if (!projectId) return;
        if (!window.ipcRenderer?.openFiles) {
            setDatasetError('Desktop file dialog is unavailable. Restart the app and try again.');
            return;
        }
        const paths = await window.ipcRenderer?.openFiles?.();
        if (!paths || paths.length === 0) return;
        setIngesting(true);
        setDatasetError(null);
        try {
            const response = await fetch(`${API_BASE_URL}/projects/${projectId}/dataset/ingest/files`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ paths })
            });
            if (!response.ok) {
                throw new Error(await readErrorMessage(response, 'Failed to import files.'));
            }
            await fetchDataset();
        } catch (error: any) {
            setDatasetError(error?.message || 'Failed to import files.');
        } finally {
            setIngesting(false);
        }
    };

    const handleIngestVideo = async () => {
        if (!projectId) return;
        if (!window.ipcRenderer?.openVideo) {
            setDatasetError('Desktop file dialog is unavailable. Restart the app and try again.');
            return;
        }
        const path = await window.ipcRenderer?.openVideo?.();
        if (!path) return;
        setIngesting(true);
        setDatasetError(null);
        try {
            const response = await fetch(`${API_BASE_URL}/projects/${projectId}/dataset/ingest/video`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path, frame_interval: frameInterval })
            });
            if (!response.ok) {
                throw new Error(await readErrorMessage(response, 'Failed to import video.'));
            }
            await fetchDataset();
        } catch (error: any) {
            setDatasetError(error?.message || 'Failed to import video.');
        } finally {
            setIngesting(false);
        }
    };

    const handleValidate = async () => {
        if (!projectId) return;
        setIsValidating(true);
        setValidationElapsedSec(0);
        setValidationLastUpdateAt(Date.now());
        setValidationReport(null);
        setScanProgress({
            phase: 'scan',
            status: 'starting',
            progress: 0,
            processed: 0,
            total: 0,
            message: 'Starting dataset validation...'
        });
        setIsDuplicateModalOpen(false);
        setDuplicateResolutionError(null);
        const response = await fetch(`${API_BASE_URL}/projects/${projectId}/dataset/validate`, { method: 'POST' });
        if (response.ok) {
            const report = await response.json().catch(() => null);
            if (report && typeof report === 'object' && 'valid_count' in report) {
                applyValidationReport(report as ValidationReport);
            }
        } else {
            const payload = await response.json().catch(() => null);
            setScanProgress({
                phase: 'validate',
                status: 'error',
                progress: 0,
                processed: 0,
                total: 0,
                message: payload?.detail || 'Validation failed. Please try again.'
            });
            setIsValidating(false);
        }
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

    const toggleDetectionLabeledClassFilter = (classId: number) => {
        setDetectionLabeledClassFilters((prev) =>
            prev.includes(classId)
                ? prev.filter((currentId) => currentId !== classId)
                : [...prev, classId]
        );
    };

    const handleAssignClass = async (classId: number) => {
        if (!projectId || selectedIds.size === 0) return;
        const ids = Array.from(selectedIds);
        const prevById = new Map(images.map((image) => [image.id, image.labels]));
        const changes = ids
            .map((id) => {
                const prevLabels = prevById.get(id) ?? [];
                const nextLabels = [classId];
                return { imageId: id, prevLabels, nextLabels };
            })
            .filter((change) => !labelsEqual(change.prevLabels, change.nextLabels));
        if (changes.length === 0) {
            setSelectedIds(new Set());
            return;
        }
        pushUndo(changes);
        const changeIds = new Set(changes.map((change) => change.imageId));
        setImages((prev) =>
            prev.map((image) =>
                changeIds.has(image.id) ? { ...image, labels: [classId] } : image
            )
        );
        const updatedAt = Date.now();
        enqueuePendingBatch(
            changes.map((change) => ({
                type: 'labels',
                imageId: change.imageId,
                labels: change.nextLabels,
                updatedAt
            }))
        );
        setSelectedIds(new Set());
    };

    const handleClearLabels = async () => {
        if (!projectId || selectedIds.size === 0) return;
        const ids = Array.from(selectedIds);
        const prevById = new Map(images.map((image) => [image.id, image.labels]));
        const changes = ids
            .map((id) => {
                const prevLabels = prevById.get(id) ?? [];
                const nextLabels: number[] = [];
                return { imageId: id, prevLabels, nextLabels };
            })
            .filter((change) => !labelsEqual(change.prevLabels, change.nextLabels));
        if (changes.length === 0) {
            setSelectedIds(new Set());
            return;
        }
        pushUndo(changes);
        const changeIds = new Set(changes.map((change) => change.imageId));
        setImages((prev) =>
            prev.map((image) =>
                changeIds.has(image.id) ? { ...image, labels: [] } : image
            )
        );
        const updatedAt = Date.now();
        enqueuePendingBatch(
            changes.map((change) => ({
                type: 'labels',
                imageId: change.imageId,
                labels: change.nextLabels,
                updatedAt
            }))
        );
        setSelectedIds(new Set());
    };

    const handleDeleteImages = async (imageIds: string[], options?: { skipConfirmation?: boolean }) => {
        if (!projectId || imageIds.length === 0) return false;
        if (!options?.skipConfirmation) {
            const confirmed = window.confirm(`Delete ${imageIds.length} image(s) from this project?`);
            if (!confirmed) return false;
        }
        const res = await fetch(`${API_BASE_URL}/projects/${projectId}/dataset/images/delete`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_ids: imageIds })
        });
        if (!res.ok) return false;
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
        return true;
    };

    const handleResolveDuplicates = async (action: DuplicateResolutionAction) => {
        if (action === 'ignore') {
            setDuplicateResolutionError(null);
            setIsDuplicateModalOpen(false);
            return;
        }
        if (!projectId || !validationReport) return;
        const groups = validationReport.duplicate_groups || [];
        if (groups.length === 0) {
            setIsDuplicateModalOpen(false);
            return;
        }

        const toDelete = new Set<string>();
        groups.forEach((group) => {
            const imagesWithId = group.images.filter((image) => typeof image.id === 'string' && Boolean(image.id));
            if (imagesWithId.length === 0) return;
            if (action === 'delete_all') {
                imagesWithId.forEach((image) => toDelete.add(String(image.id)));
                return;
            }
            const ordered = imagesWithId.slice().sort(compareDuplicateCandidates);
            const keep = ordered[0];
            imagesWithId.forEach((image) => {
                if (image.id !== keep.id) {
                    toDelete.add(String(image.id));
                }
            });
        });

        const ids = Array.from(toDelete);
        if (ids.length === 0) {
            setDuplicateResolutionError('No dataset image could be removed from these duplicate groups.');
            return;
        }

        setIsResolvingDuplicates(true);
        setDuplicateResolutionError(null);
        try {
            const deleted = await handleDeleteImages(ids, { skipConfirmation: true });
            if (!deleted) {
                setDuplicateResolutionError('Failed to remove duplicate images.');
                return;
            }
            await fetchDataset();
            setValidationReport((prev) =>
                prev
                    ? {
                        ...prev,
                        duplicate_groups_count: 0,
                        duplicate_files_count: 0,
                        duplicate_groups: []
                    }
                    : prev
            );
            setIsDuplicateModalOpen(false);
        } catch (err: any) {
            setDuplicateResolutionError(err?.message || 'Failed to resolve duplicate images.');
        } finally {
            setIsResolvingDuplicates(false);
        }
    };

    const handleAssignClassSingle = async (imageId: string, classId: number) => {
        if (!projectId) return false;
        const current = images.find((image) => image.id === imageId);
        const prevLabels = current?.labels ?? [];
        const nextLabels = [classId];
        if (labelsEqual(prevLabels, nextLabels)) return false;
        pushUndo([{ imageId, prevLabels, nextLabels }]);
        setImages((prev) =>
            prev.map((image) =>
                image.id === imageId ? { ...image, labels: [classId] } : image
            )
        );
        enqueuePending({ type: 'labels', imageId, labels: [classId], updatedAt: Date.now() });
        return true;
    };

    const handleAssignClassFromPreview = useCallback(async (imageId: string, classId: number) => {
        const index = visibleImages.findIndex((image) => image.id === imageId);
        const nextImageId = index >= 0 && index < visibleImages.length - 1
            ? visibleImages[index + 1].id
            : null;
        await handleAssignClassSingle(imageId, classId);
        setPreviewImageId(nextImageId);
    }, [handleAssignClassSingle, visibleImages]);

    const handleClearLabelsSingle = async (imageId: string) => {
        if (!projectId) return;
        const current = images.find((image) => image.id === imageId);
        const prevLabels = current?.labels ?? [];
        const nextLabels: number[] = [];
        if (labelsEqual(prevLabels, nextLabels)) return;
        pushUndo([{ imageId, prevLabels, nextLabels }]);
        setImages((prev) =>
            prev.map((image) => (image.id === imageId ? { ...image, labels: [] } : image))
        );
        enqueuePending({ type: 'labels', imageId, labels: [], updatedAt: Date.now() });
    };

    const handleDeleteImage = async (imageId: string) => {
        await handleDeleteImages([imageId]);
    };

    const openDetectionPreview = useCallback((imageId: string) => {
        setActiveImageId(imageId);
        setPreviewImageId(imageId);
    }, []);

    const goToPreviousPreviewImage = useCallback(() => {
        if (!canGoPrevPreviewImage) return;
        const target = visibleImages[previewImageIndex - 1];
        if (target) {
            setPreviewImageId(target.id);
        }
    }, [canGoPrevPreviewImage, previewImageIndex, visibleImages]);

    const goToNextPreviewImage = useCallback(() => {
        if (!canGoNextPreviewImage) return;
        const target = visibleImages[previewImageIndex + 1];
        if (target) {
            setPreviewImageId(target.id);
        }
    }, [canGoNextPreviewImage, previewImageIndex, visibleImages]);

    const handleDeleteFromPreview = useCallback(async (imageId: string) => {
        const index = visibleImages.findIndex((image) => image.id === imageId);
        const nextImageId = index >= 0 && index < visibleImages.length - 1
            ? visibleImages[index + 1].id
            : null;
        const deleted = await handleDeleteImages([imageId]);
        if (!deleted) return;
        setPreviewImageId(nextImageId);
    }, [handleDeleteImages, visibleImages]);

    const goToPreviousDetectionPreviewImage = useCallback(() => {
        if (!canGoPrevDetectionPreview) return;
        const target = detectionVisibleImages[detectionPreviewIndex - 1];
        if (target) {
            setActiveImageId(target.id);
            setPreviewImageId(target.id);
        }
    }, [canGoPrevDetectionPreview, detectionVisibleImages, detectionPreviewIndex]);

    const goToNextDetectionPreviewImage = useCallback(() => {
        if (!canGoNextDetectionPreview) return;
        const target = detectionVisibleImages[detectionPreviewIndex + 1];
        if (target) {
            setActiveImageId(target.id);
            setPreviewImageId(target.id);
        }
    }, [canGoNextDetectionPreview, detectionVisibleImages, detectionPreviewIndex]);

    const handleDeleteFromDetectionPreview = useCallback(async (imageId: string) => {
        const index = detectionVisibleImages.findIndex((image) => image.id === imageId);
        const nextImageId = index >= 0 && index < detectionVisibleImages.length - 1
            ? detectionVisibleImages[index + 1].id
            : index > 0
                ? detectionVisibleImages[index - 1].id
                : null;
        const deleted = await handleDeleteImages([imageId]);
        if (!deleted) return;
        setPreviewImageId(nextImageId);
        setActiveImageId(nextImageId);
    }, [detectionVisibleImages, handleDeleteImages]);

    const normalizeBox = (box: BoundingBox): BoundingBox => {
        const clamp = (value: number) => Math.max(0, Math.min(1, value));
        return {
            class_id: box.class_id,
            x: clamp(box.x),
            y: clamp(box.y),
            w: clamp(box.w),
            h: clamp(box.h)
        };
    };

    const updateBoxes = async (imageId: string, boxes: BoundingBox[]) => {
        if (!projectId) return;
        setImages((prev) =>
            prev.map((image) => (image.id === imageId ? { ...image, boxes } : image))
        );
        enqueuePending({ type: 'boxes', imageId, boxes, updatedAt: Date.now() });
    };

    const updateDetectionDone = async (imageId: string, detectionDone: boolean) => {
        if (!projectId) return;
        setImages((prev) =>
            prev.map((image) =>
                image.id === imageId ? { ...image, detection_done: detectionDone } : image
            )
        );
        enqueuePending({
            type: 'detection_done',
            imageId,
            detection_done: detectionDone,
            updatedAt: Date.now()
        });
    };

    const getDisplayBoxes = (boxes: BoundingBox[]) => {
        if (editingBoxIndex === null || !editingBoxDraft) return boxes;
        return boxes.map((box, index) => (index === editingBoxIndex ? editingBoxDraft : box));
    };

    const getBoxHandlePoints = (
        left: number,
        top: number,
        right: number,
        bottom: number
    ): Array<{ handle: ResizeHandle; x: number; y: number }> => {
        const midX = (left + right) / 2;
        const midY = (top + bottom) / 2;
        return [
            { handle: 'nw', x: left, y: top },
            { handle: 'n', x: midX, y: top },
            { handle: 'ne', x: right, y: top },
            { handle: 'e', x: right, y: midY },
            { handle: 'se', x: right, y: bottom },
            { handle: 's', x: midX, y: bottom },
            { handle: 'sw', x: left, y: bottom },
            { handle: 'w', x: left, y: midY }
        ];
    };

    const getResizeHandleAtPoint = (
        point: { x: number; y: number; width: number; height: number },
        box: BoundingBox
    ): ResizeHandle | null => {
        const left = box.x * point.width;
        const top = box.y * point.height;
        const right = (box.x + box.w) * point.width;
        const bottom = (box.y + box.h) * point.height;
        const hitRadius = 9;
        const handles = getBoxHandlePoints(left, top, right, bottom);
        let nearestHandle: ResizeHandle | null = null;
        let nearestDistance = Number.POSITIVE_INFINITY;
        handles.forEach((candidate) => {
            const dx = point.x - candidate.x;
            const dy = point.y - candidate.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            if (distance <= hitRadius && distance < nearestDistance) {
                nearestHandle = candidate.handle;
                nearestDistance = distance;
            }
        });
        return nearestHandle;
    };

    const getResizedBoxFromHandle = (
        box: BoundingBox,
        handle: ResizeHandle,
        nx: number,
        ny: number
    ) => {
        const left = box.x;
        const top = box.y;
        const right = box.x + box.w;
        const bottom = box.y + box.h;
        let nextLeft = left;
        let nextTop = top;
        let nextRight = right;
        let nextBottom = bottom;
        if (handle === 'nw' || handle === 'w' || handle === 'sw') {
            nextLeft = nx;
        }
        if (handle === 'ne' || handle === 'e' || handle === 'se') {
            nextRight = nx;
        }
        if (handle === 'nw' || handle === 'n' || handle === 'ne') {
            nextTop = ny;
        }
        if (handle === 'sw' || handle === 's' || handle === 'se') {
            nextBottom = ny;
        }
        return normalizeBox({
            class_id: box.class_id,
            x: Math.min(nextLeft, nextRight),
            y: Math.min(nextTop, nextBottom),
            w: Math.abs(nextRight - nextLeft),
            h: Math.abs(nextBottom - nextTop)
        });
    };

    const getBoxOverlayFill = (index: number, isEditing: boolean) => {
        const hue = Math.round((index * 137.508) % 360);
        const alpha = isEditing ? 0.26 : 0.16;
        return `hsla(${hue}, 92%, 58%, ${alpha})`;
    };

    const isPointOnBoxBorder = (
        point: { x: number; y: number; width: number; height: number },
        box: BoundingBox
    ) => {
        const left = box.x * point.width;
        const top = box.y * point.height;
        const right = (box.x + box.w) * point.width;
        const bottom = (box.y + box.h) * point.height;
        const tolerance = 6;
        const nearLeft = Math.abs(point.x - left) <= tolerance && point.y >= top - tolerance && point.y <= bottom + tolerance;
        const nearRight = Math.abs(point.x - right) <= tolerance && point.y >= top - tolerance && point.y <= bottom + tolerance;
        const nearTop = Math.abs(point.y - top) <= tolerance && point.x >= left - tolerance && point.x <= right + tolerance;
        const nearBottom = Math.abs(point.y - bottom) <= tolerance && point.x >= left - tolerance && point.x <= right + tolerance;
        return nearLeft || nearRight || nearTop || nearBottom;
    };

    const findBoxIndexAtBorder = (
        point: { x: number; y: number; width: number; height: number },
        boxes: BoundingBox[]
    ) => {
        for (let index = boxes.length - 1; index >= 0; index -= 1) {
            if (isPointOnBoxBorder(point, boxes[index])) return index;
        }
        return null;
    };

    const getDrawingSurface = () => {
        const usePreviewSurface = isDetectionPreviewOpen;
        return {
            image: usePreviewSurface ? previewImageRef.current : imageRef.current,
            canvas: usePreviewSurface ? previewCanvasRef.current : canvasRef.current
        };
    };

    const drawBoxes = (boxes: BoundingBox[], draft?: BoundingBox | null) => {
        const { canvas, image: img } = getDrawingSurface();
        if (!canvas || !img) return;
        const rect = img.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const displayBoxes = getDisplayBoxes(boxes);
        const allBoxes = draft ? [...displayBoxes, draft] : displayBoxes;
        allBoxes.forEach((box, index) => {
            const x = box.x * rect.width;
            const y = box.y * rect.height;
            const w = box.w * rect.width;
            const h = box.h * rect.height;
            const color = classColorMap.get(box.class_id) || '#2fb8ff';
            const isEditing = editingBoxIndex !== null && editingBoxIndex === index && !draft;
            ctx.fillStyle = getBoxOverlayFill(index, isEditing);
            ctx.fillRect(x, y, w, h);
            ctx.strokeStyle = color;
            ctx.lineWidth = isEditing ? 3 : 2;
            ctx.strokeRect(x, y, w, h);
            if (isEditing) {
                const handleRadius = 4;
                const handlePoints = getBoxHandlePoints(x, y, x + w, y + h);
                handlePoints.forEach((handlePoint) => {
                    ctx.beginPath();
                    ctx.fillStyle = '#ffffff';
                    ctx.arc(handlePoint.x, handlePoint.y, handleRadius, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.lineWidth = 2;
                    ctx.strokeStyle = color;
                    ctx.stroke();
                });
            }
            ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
            ctx.fillRect(x, Math.max(y - 18, 0), 86, 16);
            ctx.fillStyle = color;
            ctx.font = '11px JetBrains Mono, monospace';
            ctx.fillText(classMap.get(box.class_id) || `Class ${box.class_id}`, x + 4, Math.max(y - 6, 10));
        });
    };

    useEffect(() => {
        if (selectedImage) {
            drawBoxes(selectedImage.boxes, draftBox);
        }
    }, [selectedImage, draftBox, editingBoxIndex, editingBoxDraft, classColorMap, classMap, isDetectionPreviewOpen]);

    const getRelativePoint = (event: React.MouseEvent<HTMLDivElement, MouseEvent>, clampToImage = false) => {
        const { image: img } = getDrawingSurface();
        if (!img) return null;
        const rect = img.getBoundingClientRect();
        const isOutside =
            event.clientX < rect.left ||
            event.clientX > rect.right ||
            event.clientY < rect.top ||
            event.clientY > rect.bottom;
        if (!clampToImage && isOutside) return null;
        const x = clampToImage
            ? Math.min(Math.max(event.clientX - rect.left, 0), rect.width)
            : event.clientX - rect.left;
        const y = clampToImage
            ? Math.min(Math.max(event.clientY - rect.top, 0), rect.height)
            : event.clientY - rect.top;
        return { x, y, width: rect.width, height: rect.height };
    };

    const handleMouseDown = (event: React.MouseEvent<HTMLDivElement, MouseEvent>) => {
        if (event.button !== 0) return;
        if (!selectedImage) return;
        event.preventDefault();
        const point = getRelativePoint(event);
        if (!point) return;
        const displayBoxes = getDisplayBoxes(selectedImage.boxes);

        for (let index = displayBoxes.length - 1; index >= 0; index -= 1) {
            const handle = getResizeHandleAtPoint(point, displayBoxes[index]);
            if (handle) {
                setEditingBoxIndex(index);
                setResizeDrag({
                    boxIndex: index,
                    handle,
                    sourceBox: { ...displayBoxes[index] }
                });
                setEditingBoxDraft({ ...displayBoxes[index] });
                return;
            }
        }

        const hitBoxIndex = findBoxIndexAtBorder(point, displayBoxes);
        if (hitBoxIndex !== null) {
            setEditingBoxIndex(hitBoxIndex);
            setEditingBoxDraft(null);
            setIsDrawing(false);
            setDraftBox(null);
            return;
        }

        setEditingBoxIndex(null);
        setEditingBoxDraft(null);
        if (activeClassId === null) return;
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
        if (resizeDrag && selectedImage) {
            if ((event.buttons & 1) !== 1) {
                setResizeDrag(null);
                setEditingBoxDraft(null);
                return;
            }
            event.preventDefault();
            const point = getRelativePoint(event, true);
            if (!point) return;
            const source = selectedImage.boxes[resizeDrag.boxIndex];
            if (!source) return;
            const nx = point.x / point.width;
            const ny = point.y / point.height;
            const nextBox = getResizedBoxFromHandle(
                resizeDrag.sourceBox,
                resizeDrag.handle,
                nx,
                ny
            );
            setEditingBoxIndex(resizeDrag.boxIndex);
            setEditingBoxDraft({ ...nextBox, class_id: source.class_id });
            return;
        }

        if (!isDrawing || !draftBox) return;
        if ((event.buttons & 1) !== 1) {
            setIsDrawing(false);
            setDraftBox(null);
            return;
        }
        event.preventDefault();
        const point = getRelativePoint(event, true);
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
        if (resizeDrag && selectedImage) {
            const sourceIndex = resizeDrag.boxIndex;
            const resized = editingBoxDraft;
            setResizeDrag(null);
            if (resized && sourceIndex >= 0 && sourceIndex < selectedImage.boxes.length) {
                const nextBoxes = selectedImage.boxes.slice();
                nextBoxes[sourceIndex] = normalizeBox(resized);
                setEditingBoxDraft(null);
                await updateBoxes(selectedImage.id, nextBoxes);
            }
            return;
        }

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
        const nextBoxes = [...selectedImage.boxes, normalizeBox(draftBox)];
        setDraftBox(null);
        await updateBoxes(selectedImage.id, nextBoxes);
    };

    const handleRemoveBox = async (index: number) => {
        if (!selectedImage) return;
        const nextBoxes = selectedImage.boxes.filter((_, idx) => idx !== index);
        if (editingBoxIndex === index) {
            setEditingBoxIndex(null);
            setEditingBoxDraft(null);
        }
        await updateBoxes(selectedImage.id, nextBoxes);
    };

    const handleStartEditBox = (index: number) => {
        if (!selectedImage) return;
        const target = selectedImage.boxes[index];
        if (!target) return;
        setEditingBoxIndex(index);
        setEditingBoxDraft({ ...target });
    };

    const handleCancelEditBox = () => {
        setEditingBoxIndex(null);
        setEditingBoxDraft(null);
    };

    const handleEditBoxFieldChange = (field: keyof BoundingBox, value: string) => {
        if (!editingBoxDraft) return;
        if (field === 'class_id') {
            setEditingBoxDraft({ ...editingBoxDraft, class_id: Number(value) });
            return;
        }
        const nextValue = Number(value);
        if (Number.isNaN(nextValue)) return;
        setEditingBoxDraft({ ...editingBoxDraft, [field]: nextValue });
    };

    const handleSaveEditedBox = async () => {
        if (!selectedImage || editingBoxIndex === null || !editingBoxDraft) return;
        const nextBoxes = selectedImage.boxes.slice();
        nextBoxes[editingBoxIndex] = normalizeBox(editingBoxDraft);
        await updateBoxes(selectedImage.id, nextBoxes);
        setEditingBoxIndex(null);
        setEditingBoxDraft(null);
    };

    const goToPreviousDetectionImage = useCallback(() => {
        if (!canGoPrevDetection) return;
        const target = detectionVisibleImages[activeDetectionIndex - 1];
        if (target) {
            setActiveImageId(target.id);
        }
    }, [canGoPrevDetection, detectionVisibleImages, activeDetectionIndex]);

    const goToNextDetectionImage = useCallback(() => {
        if (!canGoNextDetection) return;
        const target = detectionVisibleImages[activeDetectionIndex + 1];
        if (target) {
            setActiveImageId(target.id);
        }
    }, [canGoNextDetection, detectionVisibleImages, activeDetectionIndex]);

    const handleDetectionSidebarResizeStart = (
        event: React.MouseEvent<HTMLButtonElement, MouseEvent>
    ) => {
        if (event.button !== 0) return;
        event.preventDefault();
        setIsResizingDetectionSidebar(true);
    };

    const handleResetDetectionSidebarWidth = () => {
        setDetectionSidebarWidth(380);
    };

    const handleToggleDetectionDone = async () => {
        if (!selectedImage) return;
        await updateDetectionDone(selectedImage.id, !selectedImage.detection_done);
    };

    const handleSaveDetectionDone = async () => {
        if (!selectedImage) return;
        const currentImageId = selectedImage.id;
        const currentIndex = detectionVisibleImages.findIndex((image) => image.id === currentImageId);
        const nextImageId = currentIndex >= 0 && currentIndex < detectionVisibleImages.length - 1
            ? detectionVisibleImages[currentIndex + 1].id
            : currentIndex > 0
                ? detectionVisibleImages[currentIndex - 1].id
                : null;

        if (!selectedImage.detection_done) {
            await updateDetectionDone(currentImageId, true);
        }

        if (previewImageId) {
            setPreviewImageId(nextImageId);
            setActiveImageId(nextImageId);
            return;
        }

        if (nextImageId && nextImageId !== currentImageId) {
            setActiveImageId(nextImageId);
        }
    };

    const totalSelected = selectedIds.size;

    useEffect(() => {
        if (!previewImageId) return;
        const handleKey = (event: KeyboardEvent) => {
            if (isEditableTarget(event.target)) return;
            if (event.key === 'Escape') {
                event.preventDefault();
                setPreviewImageId(null);
                return;
            }
            if (dataset?.task === 'classification') {
                if (event.key === 'ArrowLeft') {
                    event.preventDefault();
                    goToPreviousPreviewImage();
                    return;
                }
                if (event.key === 'ArrowRight') {
                    event.preventDefault();
                    goToNextPreviewImage();
                }
                return;
            }
            if (dataset?.task !== 'detection') return;
            if (event.key === 'ArrowLeft') {
                event.preventDefault();
                goToPreviousDetectionPreviewImage();
                return;
            }
            if (event.key === 'ArrowRight') {
                event.preventDefault();
                goToNextDetectionPreviewImage();
            }
        };
        window.addEventListener('keydown', handleKey);
        return () => window.removeEventListener('keydown', handleKey);
    }, [
        dataset?.task,
        goToNextDetectionPreviewImage,
        goToNextPreviewImage,
        goToPreviousDetectionPreviewImage,
        goToPreviousPreviewImage,
        previewImageId
    ]);

    useEffect(() => {
        if (dataset?.task !== 'classification' || classes.length === 0) return;
        const handleShortcut = (event: KeyboardEvent) => {
            if (event.metaKey || event.ctrlKey || event.altKey) return;
            if (isEditableTarget(event.target)) return;
            const digit = getShortcutDigit(event);
            if (!digit) return;
            const targetClass = classes[digit - 1];
            if (!targetClass) return;
            if (previewImageId) {
                event.preventDefault();
                void handleAssignClassFromPreview(previewImageId, targetClass.id);
                return;
            }
            if (selectedIds.size > 0) {
                event.preventDefault();
                void handleAssignClass(targetClass.id);
            }
        };
        window.addEventListener('keydown', handleShortcut);
        return () => window.removeEventListener('keydown', handleShortcut);
    }, [dataset?.task, classes, previewImageId, selectedIds, handleAssignClass, handleAssignClassFromPreview]);

    useEffect(() => {
        if (dataset?.task !== 'classification') return;
        const handleUndo = (event: KeyboardEvent) => {
            if (event.defaultPrevented || event.altKey || event.shiftKey) return;
            if (!event.ctrlKey && !event.metaKey) return;
            if (event.key.toLowerCase() !== 'z') return;
            if (isEditableTarget(event.target)) return;
            const stack = undoRef.current.slice();
            const last = stack.pop();
            if (!last) return;
            event.preventDefault();
            undoRef.current = stack;
            applyUndo(last);
        };
        window.addEventListener('keydown', handleUndo);
        return () => window.removeEventListener('keydown', handleUndo);
    }, [dataset?.task, applyUndo]);

    useEffect(() => {
        if (dataset?.task !== 'classification') return;
        const handleDeleteShortcut = (event: KeyboardEvent) => {
            if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) return;
            if (isEditableTarget(event.target)) return;
            if (event.key !== 'Delete') return;
            if (selectedIds.size === 0) return;
            event.preventDefault();
            void handleDeleteImages(Array.from(selectedIds));
        };
        window.addEventListener('keydown', handleDeleteShortcut);
        return () => window.removeEventListener('keydown', handleDeleteShortcut);
    }, [dataset?.task, selectedIds, handleDeleteImages]);

    useEffect(() => {
        if (dataset?.task !== 'detection') return;
        const handleArrowNav = (event: KeyboardEvent) => {
            if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) return;
            if (isEditableTarget(event.target)) return;
            if (previewImageId) return;
            if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
                event.preventDefault();
                goToPreviousDetectionImage();
            } else if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
                event.preventDefault();
                goToNextDetectionImage();
            }
        };
        window.addEventListener('keydown', handleArrowNav);
        return () => window.removeEventListener('keydown', handleArrowNav);
    }, [dataset?.task, goToPreviousDetectionImage, goToNextDetectionImage, previewImageId]);

    useEffect(() => {
        if (dataset?.task !== 'detection') return;
        const handleSaveDetectionShortcut = (event: KeyboardEvent) => {
            if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) return;
            if (isEditableTarget(event.target)) return;
            if (event.key.toLowerCase() !== 's') return;
            event.preventDefault();
            void handleSaveDetectionDone();
        };
        window.addEventListener('keydown', handleSaveDetectionShortcut);
        return () => window.removeEventListener('keydown', handleSaveDetectionShortcut);
    }, [dataset?.task, handleSaveDetectionDone]);

    useEffect(() => {
        if (dataset?.task !== 'detection' || classes.length === 0) return;
        const handleDetectionClassShortcut = (event: KeyboardEvent) => {
            if (event.metaKey || event.ctrlKey || event.altKey) return;
            if (isEditableTarget(event.target)) return;
            const digit = getShortcutDigit(event);
            if (!digit) return;
            const targetClass = classes[digit - 1];
            if (!targetClass) return;
            event.preventDefault();
            setActiveClassId(targetClass.id);
        };
        window.addEventListener('keydown', handleDetectionClassShortcut);
        return () => window.removeEventListener('keydown', handleDetectionClassShortcut);
    }, [dataset?.task, classes]);

    useEffect(() => {
        if (dataset?.task !== 'detection') return;
        const handleDeleteDetection = (event: KeyboardEvent) => {
            if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) return;
            if (isEditableTarget(event.target)) return;
            if (event.key !== 'Delete') return;
            if (!selectedImage) return;
            event.preventDefault();
            const hasSelectedBox =
                editingBoxIndex !== null &&
                editingBoxIndex >= 0 &&
                editingBoxIndex < selectedImage.boxes.length;
            if (hasSelectedBox) {
                void handleRemoveBox(editingBoxIndex);
                return;
            }
            void handleDeleteImage(selectedImage.id);
        };
        window.addEventListener('keydown', handleDeleteDetection);
        return () => window.removeEventListener('keydown', handleDeleteDetection);
    }, [dataset?.task, selectedImage, editingBoxIndex, handleDeleteImage, handleRemoveBox]);

    useEffect(() => {
        setSelectedIds(new Set());
        if (classificationTab !== 'labeled') {
            setLabeledClassFilter(null);
        }
        if (classificationTab !== 'unlabeled') {
            setUnlabeledSuggestionFilter(null);
        }
    }, [classificationTab]);

    useEffect(() => {
        if (unlabeledSuggestionFilter === null) return;
        const exists = unlabeledSuggestionBreakdown.some((item) => item.classId === unlabeledSuggestionFilter);
        if (!exists) {
            setUnlabeledSuggestionFilter(null);
        }
    }, [unlabeledSuggestionBreakdown, unlabeledSuggestionFilter]);

    useEffect(() => {
        setSelectedIds(new Set());
    }, [classificationAugmentedFilter]);

    useEffect(() => {
        if (!isValidating) {
            setValidationElapsedSec(0);
            return;
        }
        const startedAt = Date.now();
        const timer = window.setInterval(() => {
            setValidationElapsedSec(Math.floor((Date.now() - startedAt) / 1000));
        }, 1000);
        return () => window.clearInterval(timer);
    }, [isValidating]);

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

                    <div className="grid gap-3">
                        <Input
                            label="Frame interval (video)"
                            type="number"
                            value={frameInterval}
                            onChange={(e) => {
                                const parsed = Number(e.target.value);
                                if (!Number.isFinite(parsed)) return;
                                setFrameInterval(Math.max(1, Math.trunc(parsed)));
                            }}
                        />
                        <div className="rounded-lg border border-graphite-700/70 bg-graphite-900 px-3 py-2 text-xs text-secondary">
                            Annotation format will be resolved during training based on the selected framework/model.
                        </div>
                    </div>

                    <div className="flex flex-wrap gap-4 text-xs uppercase tracking-[0.3em] text-haze">
                        <span>Images: {imageTotal || images.length}</span>
                        <span>Classes: {dataset?.classes.length || 0}</span>
                    </div>
                    {imageTotal > IMAGE_PAGE_SIZE && images.length < imageTotal && (
                        <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-secondary">
                            <span>
                                Showing {Math.min(imageOffset + 1, imageTotal)}-
                                {Math.min(imageOffset + images.length, imageTotal)} of {imageTotal}
                            </span>
                            <div className="flex items-center gap-2">
                                <Button
                                    size="sm"
                                    variant="ghost"
                                    onClick={() => loadImages(Math.max(imageOffset - IMAGE_PAGE_SIZE, 0))}
                                    disabled={imageOffset === 0}
                                >
                                    Earlier
                                </Button>
                                <Button
                                    size="sm"
                                    variant="ghost"
                                    onClick={() => loadImages(Math.max(imageTotal - IMAGE_PAGE_SIZE, 0))}
                                    disabled={imageOffset + images.length >= imageTotal}
                                >
                                    Latest
                                </Button>
                            </div>
                        </div>
                    )}
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
                    <div className="flex flex-wrap items-center justify-between gap-3">
                        <div className="flex items-center gap-2">
                            <Loader2 size={16} className="animate-spin text-accent-primary" />
                            <h3 className="uppercase tracking-[0.28em] text-sm">Validation in progress</h3>
                        </div>
                        <div className="text-[11px] uppercase tracking-[0.3em] text-secondary">
                            Elapsed {formatElapsedSeconds(validationElapsedSec)}
                        </div>
                    </div>
                    <div className="mt-4 space-y-3">
                        <div className="text-[11px] uppercase tracking-[0.3em] text-secondary">
                            {validationPhaseLabel}
                        </div>
                        <div className="text-sm text-secondary">
                            {validationMessage}
                        </div>
                        <div className="text-xs text-secondary">
                            {validationTotal > 0
                                ? `Processed ${validationProcessed} / ${validationTotal} files`
                                : `Processed ${validationProcessed} files`}
                            {' '}· last update {validationStaleSeconds}s ago
                        </div>
                        <div className="h-2 bg-graphite-700 rounded overflow-hidden">
                            {validationProgress > 0 ? (
                                <div
                                    className="h-full bg-accent-primary transition-all duration-300"
                                    style={{ width: `${validationProgress}%` }}
                                />
                            ) : (
                                <div className="h-full w-1/3 bg-accent-primary/80 animate-pulse" />
                            )}
                        </div>
                    </div>
                </Card>
            )}

            {validationReport && (
                <div className="grid gap-6 md:grid-cols-3">
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
                            {validationReport.corrupted_files.map((f, i) => (
                                <div key={i} className="text-xs text-error font-mono">{f}</div>
                            ))}
                        </div>
                    </Card>
                    <Card glass>
                        <h3 className="mb-4 text-accent-warning">Duplicates</h3>
                        <div className="text-4xl font-bold">
                            {validationReport.duplicate_groups_count}{' '}
                            <span className="text-base text-secondary font-normal">groups</span>
                        </div>
                        <div className="mt-3 text-xs text-secondary">
                            {validationReport.duplicate_files_count} images in duplicate groups.
                        </div>
                        {validationReport.duplicate_groups_count > 0 && (
                            <div className="mt-4">
                                <Button size="sm" variant="secondary" onClick={() => setIsDuplicateModalOpen(true)}>
                                    <Copy size={14} />
                                    Resolve duplicates
                                </Button>
                            </div>
                        )}
                    </Card>
                </div>
            )}

            {isDuplicateModalOpen && validationReport && validationReport.duplicate_groups_count > 0 && (
                <div className="modal-backdrop" role="dialog" aria-modal="true">
                    <div className="modal-card max-w-[760px] max-h-[90vh] overflow-y-auto">
                        <div className="modal-header">
                            <div>
                                <div className="text-xs uppercase tracking-[0.4em] text-haze">Validate images</div>
                                <div className="text-lg font-semibold">Duplicate images found</div>
                                <div className="mt-1 text-xs text-secondary">
                                    {validationReport.duplicate_groups_count} groups and {validationReport.duplicate_files_count} duplicate images.
                                </div>
                            </div>
                            <button
                                type="button"
                                className="window-button"
                                onClick={() => handleResolveDuplicates('ignore')}
                                aria-label="Close"
                                disabled={isResolvingDuplicates}
                            >
                                <X size={14} />
                            </button>
                        </div>

                        {duplicateResolutionError && (
                            <div className="mt-3 text-xs text-error">{duplicateResolutionError}</div>
                        )}

                        <div className="mt-4 space-y-3">
                            {duplicateGroupsPreview.map((group, groupIndex) => (
                                <div
                                    key={`${group.content_hash}-${groupIndex}`}
                                    className="rounded-xl border border-graphite-700/70 bg-graphite-900/60 p-3"
                                >
                                    <div className="text-[11px] uppercase tracking-[0.3em] text-secondary">
                                        Group {groupIndex + 1} · {group.count} images
                                    </div>
                                    <div className="mt-2 space-y-1.5">
                                        {group.images.slice(0, 6).map((image, imageIndex) => (
                                            <div
                                                key={`${group.content_hash}-${image.id || image.path}-${imageIndex}`}
                                                className="flex flex-wrap items-center gap-2 text-xs text-secondary"
                                                >
                                                    <span className="font-mono text-bone">{getFileNameFromPath(image.path)}</span>
                                                    <span className="rounded-full border border-graphite-700/70 bg-graphite-900 px-2 py-0.5 text-[10px] uppercase tracking-[0.3em]">
                                                    {image.annotated ? 'Labeled' : 'Unlabeled'}
                                                </span>
                                                {!image.id && (
                                                    <span className="text-[10px] uppercase tracking-[0.3em] text-accent-warning">
                                                        not linked to dataset
                                                    </span>
                                                )}
                                            </div>
                                        ))}
                                        {group.images.length > 6 && (
                                            <div className="text-[10px] uppercase tracking-[0.3em] text-secondary">
                                                +{group.images.length - 6} images in this group
                                            </div>
                                        )}
                                    </div>
                                </div>
                            ))}
                            {hiddenDuplicateGroupsCount > 0 && (
                                <div className="text-[10px] uppercase tracking-[0.3em] text-secondary">
                                    +{hiddenDuplicateGroupsCount} groups not shown
                                </div>
                            )}
                        </div>

                        <div className="modal-actions mt-5">
                            <Button
                                size="sm"
                                variant="secondary"
                                onClick={() => handleResolveDuplicates('ignore')}
                                disabled={isResolvingDuplicates}
                            >
                                Ignore
                            </Button>
                            <Button
                                size="sm"
                                variant="primary"
                                onClick={() => handleResolveDuplicates('delete_keep_one')}
                                disabled={isResolvingDuplicates}
                            >
                                {isResolvingDuplicates ? 'Applying...' : 'Delete and keep 1'}
                            </Button>
                            <Button
                                size="sm"
                                variant="secondary"
                                className="border-accent-error/60 text-accent-error hover:text-white"
                                onClick={() => handleResolveDuplicates('delete_all')}
                                disabled={isResolvingDuplicates}
                            >
                                {isResolvingDuplicates ? 'Applying...' : 'Delete all'}
                            </Button>
                        </div>
                    </div>
                </div>
            )}

            {loading ? (
                <Card>
                    <div className="text-secondary">Loading dataset...</div>
                </Card>
            ) : datasetError ? (
                <Card className="space-y-3">
                    <div className="text-error text-sm">{datasetError}</div>
                    <div className="text-secondary text-xs">Check backend connectivity and try loading this project again.</div>
                    <div className="flex gap-2">
                        <Button variant="secondary" onClick={fetchDataset}>
                            Retry
                        </Button>
                        <Button variant="ghost" onClick={() => navigate('/projects')}>
                            Back to Projects
                        </Button>
                    </div>
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
                            {augmentationError && (
                                <div className="text-xs text-error">{augmentationError}</div>
                            )}
                            {augmentationSummary && (
                                <div className="text-xs text-accent-success">{augmentationSummary}</div>
                            )}
                            {(isAugmenting || augmentationProgress.status === 'running' || augmentationProgress.status === 'starting') && (
                                <div className="space-y-2 rounded-xl border border-graphite-700/70 bg-graphite-900/60 px-3 py-2">
                                    <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.3em] text-secondary">
                                        <span>Augmentation progress</span>
                                        <span>
                                            {augmentationProgress.processed}/{augmentationProgress.total || 0}
                                        </span>
                                    </div>
                                    <div className="h-2 rounded-full bg-graphite-700/80 overflow-hidden">
                                        <div
                                            className="h-full bg-accent-primary transition-all duration-300"
                                            style={{ width: `${Math.max(0, Math.min(100, augmentationProgress.percent))}%` }}
                                        />
                                    </div>
                                    <div className="text-[10px] uppercase tracking-[0.3em] text-secondary">
                                        {Math.round(augmentationProgress.percent)}% · generated {augmentationProgress.added}
                                    </div>
                                </div>
                            )}
                            {PRECLASSIFY_ENABLED && preclassifyError && (
                                <div className="text-xs text-error">{preclassifyError}</div>
                            )}
                            {autoLabelError && (
                                <div className="text-xs text-error">{autoLabelError}</div>
                            )}
                            {autoLabelNotice && (
                                <div className="text-xs text-accent-success">{autoLabelNotice}</div>
                            )}

                            <div className="rounded-xl border border-graphite-700/70 bg-graphite-900/50 p-3">
                                <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
                                    <div className="flex-1 grid gap-3 md:grid-cols-3">
                                        <Select
                                            label="Auto-label model"
                                            options={autoLabelModelOptions.length > 0
                                                ? autoLabelModelOptions
                                                : [{ value: '', label: autoLabelLoadingModels ? 'Loading models...' : 'No trained classification model' }]}
                                            value={autoLabelModelId}
                                            onChange={(e) => setAutoLabelModelId(e.target.value)}
                                            disabled={autoLabelLoadingModels || autoLabelModels.length === 0}
                                        />
                                        <Input
                                            label="Min confidence (0-1)"
                                            type="number"
                                            step="0.01"
                                            min="0"
                                            max="1"
                                            value={autoLabelMinConfidence}
                                            onChange={(e) => {
                                                const parsed = Number(e.target.value);
                                                if (!Number.isFinite(parsed)) return;
                                                setAutoLabelMinConfidence(Math.max(0, Math.min(1, parsed)));
                                            }}
                                            disabled={autoLabelModels.length === 0 || autoLabelRunning}
                                        />
                                        <div className="flex items-center gap-2">
                                            <Button
                                                size="sm"
                                                variant="secondary"
                                                onClick={runAutoLabelClassification}
                                                disabled={autoLabelModels.length === 0 || !autoLabelModelId || autoLabelRunning}
                                            >
                                                {autoLabelRunning ? 'Classifying...' : 'Classify unlabeled images'}
                                            </Button>
                                        </div>
                                    </div>
                                    <div className="text-[11px] uppercase tracking-[0.28em] text-secondary">
                                        Suggestions: {pendingSuggestionCount} · Selected: {selectedSuggestionCount}
                                    </div>
                                </div>

                                {(autoLabelRunning || autoLabelProgress.status === 'running' || autoLabelProgress.status === 'starting') && (
                                    <div className="mt-3 space-y-2 rounded-lg border border-graphite-700/70 bg-graphite-900/70 px-3 py-2">
                                        <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.3em] text-secondary">
                                            <span>{autoLabelProgress.message || 'Auto-classification in progress...'}</span>
                                            <span>{autoLabelProgress.processed}/{autoLabelProgress.total || 0}</span>
                                        </div>
                                        <div className="h-2 rounded-full bg-graphite-700/80 overflow-hidden">
                                            <div
                                                className="h-full bg-accent-primary transition-all duration-300"
                                                style={{ width: `${Math.max(0, Math.min(100, autoLabelProgress.progress || 0))}%` }}
                                            />
                                        </div>
                                    </div>
                                )}

                                <div className="mt-3 flex flex-wrap gap-2">
                                    <Button
                                        size="sm"
                                        variant="primary"
                                        onClick={() => handleApplyAutoLabelAction('accept')}
                                        disabled={selectedSuggestionCount === 0}
                                    >
                                        Accept selected suggestions
                                    </Button>
                                    <Button
                                        size="sm"
                                        variant="secondary"
                                        onClick={() => handleApplyAutoLabelAction('reject')}
                                        disabled={selectedSuggestionCount === 0}
                                    >
                                        Reject selected suggestions
                                    </Button>
                                    <Button
                                        size="sm"
                                        variant="ghost"
                                        onClick={() => handleApplyAutoLabelAction('clear')}
                                        disabled={selectedSuggestionCount === 0}
                                    >
                                        Clear selected suggestions
                                    </Button>
                                </div>
                                <div className="mt-2 text-[10px] uppercase tracking-[0.26em] text-secondary">
                                    Tip: double-click an image with suggestion to accept it instantly.
                                </div>
                                <div className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded-lg border border-graphite-700/70 bg-graphite-900/60 px-3 py-2 text-[11px] text-secondary">
                                    <span>
                                        Dataset versions (sampling, balancing, orientation, resolution) are configured in Training Setup.
                                    </span>
                                    <Button
                                        size="sm"
                                        variant="ghost"
                                        onClick={() => navigate(`/projects/${projectId}/train/setup`)}
                                    >
                                        Open Training Setup
                                    </Button>
                                </div>
                            </div>

                            <div className="flex flex-wrap gap-2">
                                {classes.map((cls, index) => (
                                    <Button
                                        key={cls.id}
                                        size="sm"
                                        variant="secondary"
                                        onClick={() => handleAssignClass(cls.id)}
                                        title={index < 9 ? `Shortcut ${index + 1}` : undefined}
                                    >
                                        <span className="flex items-center gap-2">
                                            <span>{cls.name}</span>
                                            {index < 9 && (
                                                <span className="rounded-full border border-graphite-700/70 bg-graphite-900 px-2 py-0.5 text-[9px] uppercase tracking-[0.3em] text-secondary leading-none">
                                                    {index + 1}
                                                </span>
                                            )}
                                        </span>
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
                                <Button
                                    size="sm"
                                    variant="secondary"
                                    onClick={() => {
                                        setAugmentationError(null);
                                        setAugmentationSummary(null);
                                        setIsAugmentationModalOpen(true);
                                    }}
                                >
                                    <Sparkles size={14} />
                                    Data Augmentation
                                </Button>
                            </div>

                            {PRECLASSIFY_ENABLED && (
                                <div className="flex flex-wrap items-end gap-3">
                                    <Select
                                        label="Pre-classify model"
                                        options={[
                                            { value: 'yolov8n-cls.pt', label: 'YOLOv8 Nano (cls)' },
                                            { value: 'yolov8s-cls.pt', label: 'YOLOv8 Small (cls)' },
                                            { value: 'yolov8m-cls.pt', label: 'YOLOv8 Medium (cls)' }
                                        ]}
                                        value={preclassifyModel}
                                        onChange={(e) => setPreclassifyModel(e.target.value)}
                                    />
                                    <Input
                                        label="Top K"
                                        type="number"
                                        value={preclassifyTopK}
                                        onChange={(e) => {
                                            const parsed = Number(e.target.value);
                                            if (!Number.isFinite(parsed)) return;
                                            setPreclassifyTopK(Math.max(1, Math.trunc(parsed)));
                                        }}
                                    />
                                    <Button
                                        variant="secondary"
                                        onClick={runPreclassify}
                                        disabled={isPreclassifying}
                                    >
                                        {isPreclassifying ? 'Pre-classifying...' : 'Pre-classify images'}
                                    </Button>
                                    <div className="text-xs text-secondary">
                                        Uses selected images or the current list.
                                    </div>
                                </div>
                            )}

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
                                        Unlabeled ({progressRemaining})
                                    </button>
                                    <button
                                        type="button"
                                        className={clsx(
                                            'tab-button',
                                            classificationTab === 'labeled' && 'tab-button-active'
                                        )}
                                        onClick={() => setClassificationTab('labeled')}
                                    >
                                            Labeled ({progressLabeled})
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
                            <div className="grid gap-3 lg:grid-cols-2">
                                <div className="space-y-2">
                                    <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.32em] text-secondary">
                                        <span>Progress</span>
                                        <span>
                                            {progressLabeled}/{progressTotal} labeled · {progressRemaining} remaining
                                        </span>
                                    </div>
                                    <div className="h-2 rounded-full bg-graphite-700/80 overflow-hidden">
                                        <div
                                            className="h-full bg-accent-primary transition-all duration-300"
                                            style={{ width: `${progressPercent}%` }}
                                        />
                                    </div>
                                </div>
                                <div className="space-y-2">
                                    <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.32em] text-secondary">
                                        <span>Class balance</span>
                                        <span>{balanceTotal} labels</span>
                                    </div>
                                    {balanceBreakdown.length === 0 ? (
                                        <div className="text-xs text-secondary">No labels yet.</div>
                                    ) : (
                                        <div className="space-y-2">
                                            {balanceBreakdown.map((item, index) => (
                                                <div
                                                    key={`balance-${item.id}`}
                                                    className="flex items-center gap-3 text-[10px] uppercase tracking-[0.32em] text-secondary"
                                                >
                                                    <span
                                                        className="h-2 w-2 rounded-full"
                                                        style={{ backgroundColor: item.color || '#2fb8ff' }}
                                                    />
                                                    <span className="min-w-[72px] max-w-[140px] truncate">
                                                        {index < 9 ? `${index + 1}. ` : ''}{item.name}
                                                    </span>
                                                    <div className="flex-1 h-2 rounded-full bg-graphite-700/80 overflow-hidden">
                                                        <div
                                                            className="h-full transition-all duration-300"
                                                            style={{
                                                                width: `${item.relative}%`,
                                                                backgroundColor: item.color || '#2fb8ff'
                                                            }}
                                                            title={`${item.name}: ${item.count}`}
                                                        />
                                                    </div>
                                                    <span className="text-bone tabular-nums">{item.count}</span>
                                                    <span className="tabular-nums">{Math.round(item.portion)}%</span>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            </div>
                            {classificationTab === 'labeled' && (
                                <div className="flex flex-wrap gap-2 text-[11px] uppercase tracking-[0.3em] text-secondary">
                                    <button
                                        type="button"
                                        className={clsx(
                                            'tab-button',
                                            labeledClassFilter === null && 'tab-button-active'
                                        )}
                                        onClick={() => setLabeledClassFilter(null)}
                                    >
                                        All ({progressLabeled})
                                    </button>
                                    {classes.map((cls) => (
                                        <button
                                            key={`count-${cls.id}`}
                                            type="button"
                                            className={clsx(
                                                'tab-button',
                                                labeledClassFilter === cls.id && 'tab-button-active'
                                            )}
                                            onClick={() => setLabeledClassFilter(cls.id)}
                                        >
                                            {cls.name} ({balanceById.get(cls.id) || 0})
                                        </button>
                                    ))}
                                </div>
                            )}
                            {classificationTab === 'unlabeled' && unlabeledSuggestionBreakdown.length > 0 && (
                                <div className="flex flex-wrap items-center gap-2 text-[11px] uppercase tracking-[0.3em] text-secondary">
                                    <span className="text-secondary">Suggestion filter</span>
                                    <button
                                        type="button"
                                        className={clsx(
                                            'tab-button',
                                            unlabeledSuggestionFilter === null && 'tab-button-active'
                                        )}
                                        onClick={() => setUnlabeledSuggestionFilter(null)}
                                    >
                                        All ({unlabeledSuggestionTotal})
                                    </button>
                                    {unlabeledSuggestionBreakdown.map((item) => (
                                        <button
                                            key={`unlabeled-suggestion-filter-${item.classId}`}
                                            type="button"
                                            className={clsx(
                                                'tab-button inline-flex items-center gap-2',
                                                unlabeledSuggestionFilter === item.classId && 'tab-button-active'
                                            )}
                                            onClick={() => setUnlabeledSuggestionFilter(item.classId)}
                                        >
                                            <span
                                                className="h-2 w-2 rounded-full"
                                                style={{ backgroundColor: item.color }}
                                            />
                                            <span className="truncate max-w-[120px]">{item.name}</span>
                                            <span className="tabular-nums">{item.count}</span>
                                        </button>
                                    ))}
                                </div>
                            )}
                            <div className="flex flex-wrap gap-2 text-[11px] uppercase tracking-[0.3em] text-secondary">
                                <button
                                    type="button"
                                    className={clsx('tab-button', classificationAugmentedFilter === 'all' && 'tab-button-active')}
                                    onClick={() => setClassificationAugmentedFilter('all')}
                                >
                                    All ({images.length})
                                </button>
                                <button
                                    type="button"
                                    className={clsx('tab-button', classificationAugmentedFilter === 'original' && 'tab-button-active')}
                                    onClick={() => setClassificationAugmentedFilter('original')}
                                >
                                    Original ({classificationOriginalCount})
                                </button>
                                <button
                                    type="button"
                                    className={clsx('tab-button', classificationAugmentedFilter === 'augmented' && 'tab-button-active')}
                                    onClick={() => setClassificationAugmentedFilter('augmented')}
                                >
                                    Augmented ({classificationAugmentedCount})
                                </button>
                            </div>

                            <div
                                ref={classificationGridRef}
                                className={clsx('image-grid', `image-grid-${thumbSize}`, 'flex-1 min-h-0')}
                                style={
                                    {
                                        gridTemplateColumns: `repeat(auto-fill, minmax(${thumbSettings.min}px, 1fr))`,
                                        '--thumb-height': `${thumbSettings.height}px`
                                    } as React.CSSProperties
                                }
                            >
                                {renderedClassificationImages.map((image, index) => {
                                    const suggestion = classificationTab === 'unlabeled'
                                        ? autoLabelSuggestionsByImage[image.id]
                                        : null;
                                    const suggestionLabel = getSuggestionLabel(suggestion);
                                    const isSelected = selectedIds.has(image.id);
                                    return (
                                    <div
                                        key={image.id}
                                        onClick={(event) => {
                                            if (event.detail > 1) return;
                                            toggleSelected(image.id);
                                        }}
                                        onDoubleClick={(event) => {
                                            event.preventDefault();
                                            event.stopPropagation();
                                            if (suggestion) {
                                                void handleAcceptAutoLabelSuggestion(image.id, suggestion.variant_id);
                                                return;
                                            }
                                            setPreviewImageId(image.id);
                                        }}
                                        onKeyDown={(event) => {
                                            if (event.key === 'Enter' || event.key === ' ') {
                                                event.preventDefault();
                                                toggleSelected(image.id);
                                            }
                                        }}
                                        role="button"
                                        tabIndex={0}
                                        className={clsx(
                                            'image-card relative',
                                            imageAspect[image.id] ? `image-card-${imageAspect[image.id]}` : null,
                                            suggestion ? 'image-card-suggested' : null,
                                            isSelected ? 'image-card-selected' : null
                                        )}
                                    >
                                        <div className="absolute left-2 top-2 rounded-full border border-graphite-700/70 bg-graphite-900 px-2 py-0.5 text-[10px] uppercase tracking-[0.3em] text-secondary">
                                            {index + 1}
                                        </div>
                                        <button
                                            type="button"
                                            className="absolute right-2 top-2 z-20 inline-flex h-6 w-6 items-center justify-center rounded-full border border-graphite-700/70 bg-graphite-900/90 text-secondary transition hover:border-accent-primary hover:text-bone"
                                            title="Open preview"
                                            onClick={(event) => {
                                                event.preventDefault();
                                                event.stopPropagation();
                                                setPreviewImageId(image.id);
                                            }}
                                        >
                                            <Search size={12} />
                                        </button>
                                        {image.is_augmented && (
                                            <div className="absolute right-10 top-2 rounded-full border border-accent-primary/70 bg-graphite-900 px-2 py-0.5 text-[10px] uppercase tracking-[0.3em] text-accent-primary">
                                                Aug
                                            </div>
                                        )}
                                        {isSelected && (
                                            <div className="absolute left-2 bottom-2 rounded-full border border-accent-primary/70 bg-graphite-900 px-2 py-0.5 text-[10px] uppercase tracking-[0.25em] text-accent-primary">
                                                Selected
                                            </div>
                                        )}
                                        {suggestion && (
                                            <div className="absolute right-2 bottom-2 flex max-w-[calc(100%-16px)] items-center gap-1 rounded-full border border-accent-success/70 bg-graphite-900 px-1.5 py-0.5 text-[10px] text-accent-success">
                                                <span className="max-w-[96px] truncate uppercase tracking-[0.18em]" title={suggestionLabel}>
                                                    {suggestionLabel}
                                                </span>
                                                <span className="rounded-full border border-accent-success/60 px-1 py-[1px] uppercase tracking-[0.16em] tabular-nums">
                                                    {(suggestion.confidence * 100).toFixed(0)}%
                                                </span>
                                            </div>
                                        )}
                                        <img
                                            src={getImageThumbnailSrc(image, classificationThumbPixelSize)}
                                            alt={image.id}
                                            className="thumb-image"
                                            loading="lazy"
                                            decoding="async"
                                            onLoad={(event) => handleThumbLoad(image.id, event)}
                                        />
                                        <div className="mt-2 text-xs text-secondary">
                                            {image.labels.length > 0
                                                ? image.labels.map((id) => classMap.get(id) || id).join(', ')
                                                : 'Unlabeled'}
                                        </div>
                                        {suggestion && (
                                            <div className="mt-1 flex min-w-0 items-center gap-1 text-[10px] text-accent-success">
                                                <span className="uppercase tracking-[0.24em] text-accent-success/85">Suggested</span>
                                                <span className="truncate uppercase tracking-[0.18em]" title={suggestionLabel}>
                                                    {suggestionLabel}
                                                </span>
                                            </div>
                                        )}
                                        {PRECLASSIFY_ENABLED && preclassifyResults[image.id]?.length ? (
                                            <div className="mt-2 flex flex-wrap gap-2 text-[10px] uppercase tracking-[0.3em] text-secondary">
                                                {preclassifyResults[image.id].map((prediction) => (
                                                    <button
                                                        key={`${image.id}-${prediction.class_id}`}
                                                        type="button"
                                                        className="border border-graphite-700/70 bg-graphite-900 px-2 py-1 hover:border-accent-primary"
                                                        onClick={(event) => {
                                                            event.stopPropagation();
                                                            void handleAcceptPrelabel(image.id, prediction);
                                                        }}
                                                    >
                                                        {prediction.class_name} {(prediction.confidence * 100).toFixed(0)}%
                                                    </button>
                                                ))}
                                            </div>
                                        ) : null}
                                    </div>
                                );
                                })}
                                {canLoadMoreClassification && (
                                    <div
                                        ref={classificationSentinelRef}
                                        className="col-span-full flex items-center justify-center rounded-lg border border-graphite-700/60 bg-graphite-900/50 px-3 py-2 text-[10px] uppercase tracking-[0.3em] text-secondary"
                                    >
                                        <Loader2 size={12} className="mr-2 animate-spin" />
                                        Loading more thumbnails...
                                    </div>
                                )}
                            </div>
                        </Card>
                    )}

                    {isAugmentationModalOpen && dataset?.task === 'classification' && (
                        <div className="modal-backdrop" role="dialog" aria-modal="true">
                            <div className="modal-card max-h-[90vh] overflow-y-auto">
                                <div className="modal-header">
                                    <div>
                                        <div className="text-xs uppercase tracking-[0.4em] text-haze">Classification</div>
                                        <div className="text-lg font-semibold">Data Augmentation</div>
                                        <div className="mt-1 text-xs text-secondary">
                                            Using Ultralytics YOLO classification augmentations.
                                        </div>
                                    </div>
                                    <button
                                        type="button"
                                        className="window-button"
                                        onClick={() => setIsAugmentationModalOpen(false)}
                                        aria-label="Close"
                                        disabled={isAugmenting}
                                    >
                                        <X size={14} />
                                    </button>
                                </div>

                                <div className="mt-4 grid gap-3 sm:grid-cols-2">
                                    <Select
                                        label="Target scope"
                                        options={[
                                            { value: 'all', label: 'All images' },
                                            { value: 'labeled', label: 'Only labeled' },
                                            { value: 'unlabeled', label: 'Only unlabeled' }
                                        ]}
                                        value={augmentationConfig.target_scope}
                                        onChange={(event) =>
                                            updateAugmentationConfig(
                                                'target_scope',
                                                event.target.value as ClassificationAugmentationConfig['target_scope']
                                            )
                                        }
                                    />
                                    <Input
                                        label="Copies per image"
                                        type="number"
                                        min={1}
                                        value={augmentationConfig.copies_per_image}
                                        onChange={(event) => {
                                            const parsed = Number(event.target.value);
                                            if (!Number.isFinite(parsed)) return;
                                            updateAugmentationConfig('copies_per_image', Math.max(1, Math.trunc(parsed)));
                                        }}
                                    />
                                    <Input
                                        label="Output size (px)"
                                        type="number"
                                        min={32}
                                        value={augmentationConfig.output_size}
                                        onChange={(event) => {
                                            const parsed = Number(event.target.value);
                                            if (!Number.isFinite(parsed)) return;
                                            updateAugmentationConfig('output_size', Math.max(32, Math.trunc(parsed)));
                                        }}
                                    />
                                    <Input
                                        label="Apply percent"
                                        type="number"
                                        min={0}
                                        max={100}
                                        step="1"
                                        value={augmentationConfig.apply_percent}
                                        onChange={(event) => {
                                            const parsed = Number(event.target.value);
                                            if (!Number.isFinite(parsed)) return;
                                            updateAugmentationConfig('apply_percent', clampPercent(parsed));
                                        }}
                                    />
                                    <Select
                                        label="Auto augment policy"
                                        options={[
                                            { value: '', label: 'None' },
                                            { value: 'randaugment', label: 'RandAugment' },
                                            { value: 'augmix', label: 'AugMix' },
                                            { value: 'autoaugment', label: 'AutoAugment' }
                                        ]}
                                        value={augmentationConfig.auto_augment}
                                        onChange={(event) =>
                                            updateAugmentationConfig(
                                                'auto_augment',
                                                (event.target.value as ClassificationAugmentationConfig['auto_augment']) || ''
                                            )
                                        }
                                    />
                                </div>

                                <div className="mt-4 grid gap-3 sm:grid-cols-2">
                                    <Input
                                        label="Horizontal flip %"
                                        type="number"
                                        min={0}
                                        max={100}
                                        value={augmentationConfig.horizontal_flip_percent}
                                        onChange={(event) => {
                                            const parsed = Number(event.target.value);
                                            if (!Number.isFinite(parsed)) return;
                                            updateAugmentationConfig('horizontal_flip_percent', clampPercent(parsed));
                                        }}
                                    />
                                    <Input
                                        label="Vertical flip %"
                                        type="number"
                                        min={0}
                                        max={100}
                                        value={augmentationConfig.vertical_flip_percent}
                                        onChange={(event) => {
                                            const parsed = Number(event.target.value);
                                            if (!Number.isFinite(parsed)) return;
                                            updateAugmentationConfig('vertical_flip_percent', clampPercent(parsed));
                                        }}
                                    />
                                    <Input
                                        label="HSV Hue %"
                                        type="number"
                                        min={0}
                                        max={100}
                                        value={augmentationConfig.hsv_h_percent}
                                        onChange={(event) => {
                                            const parsed = Number(event.target.value);
                                            if (!Number.isFinite(parsed)) return;
                                            updateAugmentationConfig('hsv_h_percent', clampPercent(parsed));
                                        }}
                                    />
                                    <Input
                                        label="HSV Saturation %"
                                        type="number"
                                        min={0}
                                        max={100}
                                        value={augmentationConfig.hsv_s_percent}
                                        onChange={(event) => {
                                            const parsed = Number(event.target.value);
                                            if (!Number.isFinite(parsed)) return;
                                            updateAugmentationConfig('hsv_s_percent', clampPercent(parsed));
                                        }}
                                    />
                                    <Input
                                        label="HSV Value %"
                                        type="number"
                                        min={0}
                                        max={100}
                                        value={augmentationConfig.hsv_v_percent}
                                        onChange={(event) => {
                                            const parsed = Number(event.target.value);
                                            if (!Number.isFinite(parsed)) return;
                                            updateAugmentationConfig('hsv_v_percent', clampPercent(parsed));
                                        }}
                                    />
                                    <Input
                                        label="Random erasing %"
                                        type="number"
                                        min={0}
                                        max={100}
                                        value={augmentationConfig.erasing_percent}
                                        onChange={(event) => {
                                            const parsed = Number(event.target.value);
                                            if (!Number.isFinite(parsed)) return;
                                            updateAugmentationConfig('erasing_percent', clampPercent(parsed));
                                        }}
                                    />
                                </div>

                                <label className="mt-4 inline-flex items-center gap-2 text-xs uppercase tracking-[0.3em] text-secondary">
                                    <input
                                        type="checkbox"
                                        checked={augmentationConfig.force_color_jitter}
                                        onChange={(event) =>
                                            updateAugmentationConfig('force_color_jitter', event.target.checked)
                                        }
                                    />
                                    Force color jitter with auto-augment
                                </label>

                                <div className="mt-4 text-xs text-secondary">
                                    Target images: {selectedIds.size > 0
                                        ? augmentationSelectedScopeCount
                                        : augmentationScopeCount}
                                    {' '}({selectedIds.size > 0 ? 'selected + scope filter' : 'scope filter'})
                                </div>
                                {(isAugmenting || augmentationProgress.total > 0) && (
                                    <div className="mt-3 space-y-2">
                                        <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.3em] text-secondary">
                                            <span>Progress</span>
                                            <span>
                                                {augmentationProgress.processed}/{augmentationProgress.total || 0}
                                            </span>
                                        </div>
                                        <div className="h-2 rounded-full bg-graphite-700/80 overflow-hidden">
                                            <div
                                                className="h-full bg-accent-primary transition-all duration-300"
                                                style={{ width: `${Math.max(0, Math.min(100, augmentationProgress.percent))}%` }}
                                            />
                                        </div>
                                        <div className="text-[10px] uppercase tracking-[0.3em] text-secondary">
                                            {Math.round(augmentationProgress.percent)}% · generated {augmentationProgress.added}
                                        </div>
                                    </div>
                                )}
                                {augmentationError && (
                                    <div className="mt-2 text-xs text-error">{augmentationError}</div>
                                )}
                                <div className="modal-actions mt-5">
                                    <Button
                                        size="sm"
                                        variant="secondary"
                                        onClick={() => setIsAugmentationModalOpen(false)}
                                        disabled={isAugmenting}
                                    >
                                        Cancel
                                    </Button>
                                    <Button
                                        size="sm"
                                        variant="primary"
                                        onClick={runClassificationAugmentation}
                                        disabled={isAugmenting}
                                    >
                                        {isAugmenting ? 'Augmenting...' : 'Run Augmentation'}
                                    </Button>
                                </div>
                            </div>
                        </div>
                    )}

                    {previewImage && dataset?.task === 'classification' && (
                        <div className="modal-backdrop" role="dialog" aria-modal="true">
                            <div className="modal-card">
                                <div className="modal-header">
                                    <div>
                                        <div className="text-xs uppercase tracking-[0.4em] text-haze">Preview</div>
                                        <div className="text-lg font-semibold">{previewImage.id}</div>
                                        {previewImage.is_augmented && (
                                            <div className="text-xs uppercase tracking-[0.3em] text-accent-primary mt-1">
                                                Augmented from {previewImage.augmentation_source_id || 'unknown'}
                                            </div>
                                        )}
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
                                <div className="modal-image-frame">
                                    <img
                                        src={getImageSrc(previewImage)}
                                        alt={previewImage.id}
                                        className="modal-image"
                                    />
                                    <button
                                        type="button"
                                        className={clsx('preview-nav-button preview-nav-prev', !canGoPrevPreviewImage && 'preview-nav-button-disabled')}
                                        onClick={goToPreviousPreviewImage}
                                        disabled={!canGoPrevPreviewImage}
                                        title="Previous image (Left Arrow)"
                                        aria-label="Previous image"
                                    >
                                        <ChevronLeft size={18} />
                                    </button>
                                    <button
                                        type="button"
                                        className={clsx('preview-nav-button preview-nav-next', !canGoNextPreviewImage && 'preview-nav-button-disabled')}
                                        onClick={goToNextPreviewImage}
                                        disabled={!canGoNextPreviewImage}
                                        title="Next image (Right Arrow)"
                                        aria-label="Next image"
                                    >
                                        <ChevronRight size={18} />
                                    </button>
                                </div>
                                {PRECLASSIFY_ENABLED && preclassifyResults[previewImage.id]?.length ? (
                                    <div className="modal-labels">
                                        {preclassifyResults[previewImage.id].map((prediction) => (
                                            <Button
                                                key={`preview-${prediction.class_id}`}
                                                size="sm"
                                                variant="secondary"
                                                onClick={() => handleAcceptPrelabel(previewImage.id, prediction)}
                                            >
                                                {prediction.class_name} {(prediction.confidence * 100).toFixed(0)}%
                                            </Button>
                                        ))}
                                    </div>
                                ) : null}
                                <div className="modal-labels">
                                    {classes.map((cls, index) => (
                                        <Button
                                            key={cls.id}
                                            size="sm"
                                            variant="secondary"
                                            onClick={() => handleAssignClassFromPreview(previewImage.id, cls.id)}
                                            className={clsx(
                                                previewImage.labels.includes(cls.id) && 'border-accent-primary text-white'
                                            )}
                                            title={index < 9 ? `Shortcut ${index + 1}` : undefined}
                                        >
                                            <span className="flex items-center gap-2">
                                                <span>{cls.name}</span>
                                                {index < 9 && (
                                                    <span className="rounded-full border border-graphite-700/70 bg-graphite-900 px-2 py-0.5 text-[9px] uppercase tracking-[0.3em] text-secondary leading-none">
                                                        {index + 1}
                                                    </span>
                                                )}
                                            </span>
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
                                        onClick={() => handleDeleteFromPreview(previewImage.id)}
                                    >
                                        Delete image
                                    </Button>
                                </div>
                            </div>
                        </div>
                    )}

                    {detectionPreviewImage && dataset?.task === 'detection' && (
                        <div className="modal-backdrop" role="dialog" aria-modal="true">
                            <div className="modal-card max-w-[1320px] max-h-[94vh] overflow-y-auto">
                                <div className="modal-header">
                                    <div>
                                        <div className="text-xs uppercase tracking-[0.4em] text-haze">Detection preview</div>
                                        <div className="text-lg font-semibold">{detectionPreviewImage.id}</div>
                                        <div className="mt-1 text-[11px] uppercase tracking-[0.28em] text-secondary">
                                            Draw boxes in this centered view.
                                        </div>
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

                                <div className="mt-4 flex flex-wrap items-center justify-between gap-2">
                                    <div className="text-xs uppercase tracking-[0.3em] text-secondary">
                                        Image {detectionPreviewIndex >= 0 ? detectionPreviewIndex + 1 : 0} / {detectionVisibleImages.length}
                                    </div>
                                    <div className="flex flex-wrap items-center gap-2">
                                        <Select
                                            label="Active class"
                                            options={classes.map((cls) => ({ value: String(cls.id), label: cls.name }))}
                                            value={activeClassId !== null ? String(activeClassId) : ''}
                                            onChange={(event) => setActiveClassId(Number(event.target.value))}
                                        />
                                        <Button
                                            size="sm"
                                            variant={detectionPreviewImage.detection_done ? 'secondary' : 'primary'}
                                            className={detectionPreviewImage.detection_done ? 'border-accent-success/60 text-accent-success' : undefined}
                                            onClick={() => void updateDetectionDone(detectionPreviewImage.id, !detectionPreviewImage.detection_done)}
                                        >
                                            {detectionPreviewImage.detection_done ? 'Mark pending' : 'Mark done (S)'}
                                        </Button>
                                        <Button
                                            size="sm"
                                            variant="secondary"
                                            className="border-accent-error/60 text-accent-error hover:text-white"
                                            onClick={() => void handleDeleteFromDetectionPreview(detectionPreviewImage.id)}
                                        >
                                            Delete image
                                        </Button>
                                    </div>
                                </div>

                                <div className="modal-image-frame">
                                    <div
                                        className="relative flex min-h-[620px] items-center justify-center border-2 border-graphite-700 bg-graphite-900 p-3"
                                        onMouseDown={handleMouseDown}
                                        onMouseMove={handleMouseMove}
                                        onMouseUp={handleMouseUp}
                                        onMouseLeave={() => {
                                            if (!isDrawing) return;
                                            setIsDrawing(false);
                                            setDraftBox(null);
                                        }}
                                    >
                                        {selectedImage && selectedImage.id === detectionPreviewImage.id ? (
                                            <div className="relative inline-block max-w-full">
                                                <img
                                                    ref={previewImageRef}
                                                    src={getImageSrc(detectionPreviewImage)}
                                                    alt={detectionPreviewImage.id}
                                                    className="max-h-[76vh] max-w-full select-none object-contain"
                                                    draggable={false}
                                                    onLoad={(event) => {
                                                        handleSelectedImageLoad(event);
                                                        drawBoxes(selectedImage.boxes, draftBox);
                                                    }}
                                                />
                                                <canvas ref={previewCanvasRef} className="absolute left-0 top-0 pointer-events-none" />
                                                <button
                                                    type="button"
                                                    className={clsx('preview-nav-button preview-nav-prev', !canGoPrevDetectionPreview && 'preview-nav-button-disabled')}
                                                    onClick={goToPreviousDetectionPreviewImage}
                                                    disabled={!canGoPrevDetectionPreview}
                                                    title="Previous image (Left Arrow)"
                                                    aria-label="Previous image"
                                                >
                                                    <ChevronLeft size={18} />
                                                </button>
                                                <button
                                                    type="button"
                                                    className={clsx('preview-nav-button preview-nav-next', !canGoNextDetectionPreview && 'preview-nav-button-disabled')}
                                                    onClick={goToNextDetectionPreviewImage}
                                                    disabled={!canGoNextDetectionPreview}
                                                    title="Next image (Right Arrow)"
                                                    aria-label="Next image"
                                                >
                                                    <ChevronRight size={18} />
                                                </button>
                                            </div>
                                        ) : (
                                            <div className="text-secondary text-sm">Loading selected image...</div>
                                        )}
                                    </div>
                                </div>

                                {selectedImage && selectedImage.id === detectionPreviewImage.id && (
                                    <div className="mt-4 space-y-2">
                                        <div className="flex items-center justify-between text-xs uppercase tracking-[0.3em] text-haze">
                                            <span>Boxes</span>
                                            <span className="text-secondary">
                                                {selectedImage.id} · {selectedImage.detection_done ? 'Done' : 'Pending'}
                                            </span>
                                        </div>
                                        {selectedImage.boxes.map((box, index) => (
                                            <div
                                                key={`preview-detection-box-${selectedImage.id}-${index}`}
                                                className={clsx(
                                                    'flex items-center justify-between border-2 px-2 py-1 text-xs',
                                                    editingBoxIndex === index ? 'border-accent-primary' : 'border-graphite-700'
                                                )}
                                            >
                                                <span>{classMap.get(box.class_id) || `Class ${box.class_id}`}</span>
                                                <div className="flex items-center gap-3">
                                                    <button onClick={() => handleStartEditBox(index)} className="text-accent-primary">Edit</button>
                                                    <button onClick={() => handleRemoveBox(index)} className="text-error">Remove</button>
                                                </div>
                                            </div>
                                        ))}
                                        {selectedImage.boxes.length === 0 && (
                                            <div className="text-xs text-secondary">No boxes yet.</div>
                                        )}
                                        {editingBoxDraft && (
                                            <div className="mt-3 rounded-xl border border-accent-primary/40 bg-graphite-900 p-3 space-y-3">
                                                <div className="text-xs uppercase tracking-[0.3em] text-secondary">
                                                    Edit box #{(editingBoxIndex ?? 0) + 1}
                                                </div>
                                                <div className="grid gap-2 sm:grid-cols-2">
                                                    <Select
                                                        label="Class"
                                                        options={classes.map((cls) => ({
                                                            value: String(cls.id),
                                                            label: cls.name
                                                        }))}
                                                        value={String(editingBoxDraft.class_id)}
                                                        onChange={(event) => handleEditBoxFieldChange('class_id', event.target.value)}
                                                    />
                                                    <Input
                                                        label="X"
                                                        type="number"
                                                        min={0}
                                                        max={1}
                                                        step="0.001"
                                                        value={editingBoxDraft.x}
                                                        onChange={(event) => handleEditBoxFieldChange('x', event.target.value)}
                                                    />
                                                    <Input
                                                        label="Y"
                                                        type="number"
                                                        min={0}
                                                        max={1}
                                                        step="0.001"
                                                        value={editingBoxDraft.y}
                                                        onChange={(event) => handleEditBoxFieldChange('y', event.target.value)}
                                                    />
                                                    <Input
                                                        label="Width"
                                                        type="number"
                                                        min={0}
                                                        max={1}
                                                        step="0.001"
                                                        value={editingBoxDraft.w}
                                                        onChange={(event) => handleEditBoxFieldChange('w', event.target.value)}
                                                    />
                                                    <Input
                                                        label="Height"
                                                        type="number"
                                                        min={0}
                                                        max={1}
                                                        step="0.001"
                                                        value={editingBoxDraft.h}
                                                        onChange={(event) => handleEditBoxFieldChange('h', event.target.value)}
                                                    />
                                                </div>
                                                <div className="flex items-center justify-end gap-2">
                                                    <Button size="sm" variant="secondary" onClick={handleCancelEditBox}>
                                                        Cancel
                                                    </Button>
                                                    <Button size="sm" variant="primary" onClick={handleSaveEditedBox}>
                                                        Save box
                                                    </Button>
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {dataset && dataset.task === 'detection' && (
                        <Card className="space-y-4">
                            <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                                <div>
                                    <h3 className="text-lg font-semibold">Detection labeling</h3>
                                    <p className="text-secondary text-sm">
                                        Select a thumbnail, then draw bounding boxes on the centered image.
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

                            <div
                                ref={detectionLayoutRef}
                                className="grid grid-cols-1 gap-4 xl:grid-cols-[var(--detection-sidebar-width)_12px_minmax(0,1fr)]"
                                style={
                                    {
                                        '--detection-sidebar-width': `${detectionSidebarWidth}px`
                                    } as React.CSSProperties
                                }
                            >
                                <div className="space-y-3 min-w-0">
                                    <div className="flex flex-wrap gap-2">
                                        <button
                                            type="button"
                                            className={clsx(
                                                'tab-button',
                                                detectionTab === 'unlabeled' && 'tab-button-active'
                                            )}
                                            onClick={() => setDetectionTab('unlabeled')}
                                        >
                                            Unlabeled ({detectionRemaining})
                                        </button>
                                        <button
                                            type="button"
                                            className={clsx(
                                                'tab-button',
                                                detectionTab === 'labeled' && 'tab-button-active'
                                            )}
                                            onClick={() => setDetectionTab('labeled')}
                                        >
                                            Labeled ({detectionDone})
                                        </button>
                                    </div>
                                    {detectionTab === 'labeled' && (
                                        <div className="space-y-2 rounded-xl border border-graphite-700/70 bg-graphite-900/60 p-3">
                                            <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.3em] text-secondary">
                                                <span>Class filter</span>
                                                <button
                                                    type="button"
                                                    className="text-secondary transition hover:text-bone disabled:cursor-not-allowed disabled:opacity-50"
                                                    onClick={() => setDetectionLabeledClassFilters([])}
                                                    disabled={detectionLabeledClassFilters.length === 0}
                                                >
                                                    Clear
                                                </button>
                                            </div>
                                            <div className="flex flex-wrap gap-2">
                                                <button
                                                    type="button"
                                                    className={clsx(
                                                        'inline-flex items-center gap-2 rounded-full border px-2 py-1 text-[10px] uppercase tracking-[0.3em] transition',
                                                        detectionLabeledClassFilters.length === 0
                                                            ? 'border-accent-primary bg-accent-primary/10 text-bone'
                                                            : 'border-graphite-700 text-secondary hover:text-bone'
                                                    )}
                                                    onClick={() => setDetectionLabeledClassFilters([])}
                                                >
                                                    All
                                                    <span className="tabular-nums">{detectionDone}</span>
                                                </button>
                                                {classes.map((cls) => {
                                                    const isSelected = detectionLabeledClassFilters.includes(cls.id);
                                                    const classCount = detectionLabeledClassImageCount.get(cls.id) || 0;
                                                    return (
                                                        <button
                                                            key={`detection-filter-${cls.id}`}
                                                            type="button"
                                                            className={clsx(
                                                                'inline-flex items-center gap-2 rounded-full border px-2 py-1 text-[10px] uppercase tracking-[0.3em] transition',
                                                                isSelected
                                                                    ? 'border-accent-primary bg-accent-primary/10 text-bone'
                                                                    : 'border-graphite-700 text-secondary hover:text-bone'
                                                            )}
                                                            onClick={() => toggleDetectionLabeledClassFilter(cls.id)}
                                                        >
                                                            <span
                                                                className="h-2 w-2 rounded-full"
                                                                style={{ backgroundColor: cls.color || '#2fb8ff' }}
                                                            />
                                                            <span className="truncate max-w-[90px]">{cls.name}</span>
                                                            <span className="tabular-nums text-bone">{classCount}</span>
                                                        </button>
                                                    );
                                                })}
                                            </div>
                                            <div className="text-[10px] uppercase tracking-[0.3em] text-secondary">
                                                {detectionLabeledClassFilters.length === 0
                                                    ? `Showing ${detectionVisibleImages.length} labeled images`
                                                    : `Showing ${detectionVisibleImages.length} images matching selected classes (OR)`}
                                            </div>
                                        </div>
                                    )}

                                    <div className="space-y-3">
                                        <div className="space-y-2">
                                            <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.32em] text-secondary">
                                                <span>Labeled</span>
                                                <span>{detectionDone}/{detectionTotal} images</span>
                                            </div>
                                            <div className="h-2 rounded-full bg-graphite-700/80 overflow-hidden">
                                                <div
                                                    className="h-full bg-accent-success transition-all duration-300"
                                                    style={{ width: `${detectionProgressPercent}%` }}
                                                />
                                            </div>
                                        </div>
                                        <div className="space-y-2">
                                            <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.32em] text-secondary">
                                                <span>Unlabeled</span>
                                                <span>{detectionRemaining}/{detectionTotal} images</span>
                                            </div>
                                            <div className="h-2 rounded-full bg-graphite-700/80 overflow-hidden">
                                                <div
                                                    className="h-full bg-accent-primary transition-all duration-300"
                                                    style={{ width: `${detectionRemainingPercent}%` }}
                                                />
                                            </div>
                                        </div>
                                    </div>

                                    <div className="space-y-2">
                                        <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.32em] text-secondary">
                                            <span>Class balance</span>
                                            <span>{detectionBalanceTotal} boxes</span>
                                        </div>
                                        {detectionBalanceBreakdown.length === 0 ? (
                                            <div className="text-xs text-secondary">No classes yet.</div>
                                        ) : (
                                            <div className="space-y-2">
                                                {detectionBalanceBreakdown.map((item, index) => (
                                                    <div
                                                        key={`detection-balance-${item.id}`}
                                                        className="flex items-center gap-3 text-[10px] uppercase tracking-[0.32em] text-secondary"
                                                    >
                                                        <span
                                                            className="h-2 w-2 rounded-full"
                                                            style={{ backgroundColor: item.color || '#2fb8ff' }}
                                                        />
                                                        <span className="min-w-[72px] max-w-[130px] truncate">
                                                            {index < 9 ? `${index + 1}. ` : ''}{item.name}
                                                        </span>
                                                        <div className="flex-1 h-2 rounded-full bg-graphite-700/80 overflow-hidden">
                                                            <div
                                                                className="h-full transition-all duration-300"
                                                                style={{
                                                                    width: `${item.relative}%`,
                                                                    backgroundColor: item.color || '#2fb8ff'
                                                                }}
                                                                title={`${item.name}: ${item.count}`}
                                                            />
                                                        </div>
                                                        <span className="text-bone tabular-nums">{item.count}</span>
                                                        <span className="tabular-nums">{Math.round(item.portion)}%</span>
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                    </div>

                                    <div
                                        ref={detectionGridRef}
                                        className="grid gap-2 max-h-[460px] overflow-y-auto pr-1"
                                        style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(210px, 1fr))' }}
                                    >
                                        {renderedDetectionImages.map((image, index) => (
                                            <div
                                                key={image.id}
                                                onClick={() => setActiveImageId(image.id)}
                                                onKeyDown={(event) => {
                                                    if (event.key === 'Enter' || event.key === ' ') {
                                                        event.preventDefault();
                                                        setActiveImageId(image.id);
                                                    }
                                                }}
                                                role="button"
                                                tabIndex={0}
                                                className={clsx(
                                                    'image-card relative text-left',
                                                    activeImageId === image.id && 'image-card-selected'
                                                )}
                                            >
                                                <span className="absolute left-2 top-2 rounded-full border border-graphite-700/70 bg-graphite-900 px-2 py-0.5 text-[10px] uppercase tracking-[0.3em] text-secondary">
                                                    {index + 1}
                                                </span>
                                                <button
                                                    type="button"
                                                    className="absolute right-2 top-2 z-20 inline-flex h-6 w-6 items-center justify-center rounded-full border border-graphite-700/70 bg-graphite-900/90 text-secondary transition hover:border-accent-primary hover:text-bone"
                                                    title="Open preview"
                                                    onClick={(event) => {
                                                        event.preventDefault();
                                                        event.stopPropagation();
                                                        openDetectionPreview(image.id);
                                                    }}
                                                >
                                                    <Search size={12} />
                                                </button>
                                                {image.detection_done && (
                                                    <span className="absolute right-10 top-2 rounded-full border border-accent-success/50 bg-graphite-900 px-2 py-0.5 text-[10px] uppercase tracking-[0.3em] text-accent-success">
                                                        Done
                                                    </span>
                                                )}
                                                <img
                                                    src={getImageThumbnailSrc(image, 256)}
                                                    alt={image.id}
                                                    className="h-24 w-full object-cover bg-graphite-950/70"
                                                    loading="lazy"
                                                    decoding="async"
                                                />
                                                <div className="mt-2 flex items-center justify-between gap-2 text-[10px] uppercase tracking-[0.26em] text-secondary">
                                                    <span className="truncate">{image.id}</span>
                                                    <span>{image.boxes.length}</span>
                                                </div>
                                            </div>
                                        ))}
                                        {canLoadMoreDetection && (
                                            <div
                                                ref={detectionSentinelRef}
                                                className="col-span-2 flex items-center justify-center rounded-xl border border-graphite-700/70 bg-graphite-900/50 px-3 py-3 text-[10px] uppercase tracking-[0.3em] text-secondary"
                                            >
                                                <Loader2 size={12} className="mr-2 animate-spin" />
                                                Loading more thumbnails...
                                            </div>
                                        )}
                                        {detectionVisibleImages.length === 0 && (
                                            <div className="col-span-2 rounded-xl border border-graphite-700/70 bg-graphite-900 px-3 py-6 text-center text-xs uppercase tracking-[0.3em] text-secondary">
                                                No images in this tab.
                                            </div>
                                        )}
                                    </div>
                                </div>

                                <div className="relative hidden xl:flex items-stretch">
                                    <button
                                        type="button"
                                        className={clsx(
                                            'group relative w-full cursor-col-resize rounded-full border border-graphite-700/70 bg-graphite-900/70 transition-colors',
                                            isResizingDetectionSidebar && 'border-accent-primary/70 bg-graphite-800'
                                        )}
                                        onMouseDown={handleDetectionSidebarResizeStart}
                                        onDoubleClick={handleResetDetectionSidebarWidth}
                                        aria-label="Resize thumbnail panel"
                                        title="Drag to resize panels"
                                    >
                                        <span className="absolute inset-y-0 left-1/2 -translate-x-1/2 flex items-center text-secondary transition-colors group-hover:text-bone">
                                            <GripVertical size={14} />
                                        </span>
                                    </button>
                                </div>

                                <div className="space-y-4 min-w-0">
                                    <div className="flex flex-wrap items-center justify-between gap-2">
                                        <div className="text-xs uppercase tracking-[0.3em] text-secondary">
                                            Image {activeDetectionIndex >= 0 ? activeDetectionIndex + 1 : 0} / {detectionVisibleImages.length}
                                        </div>
                                        <div className="flex items-center gap-2">
                                            {selectedImage && (
                                                <Button
                                                    size="sm"
                                                    variant={selectedImage.detection_done ? 'secondary' : 'primary'}
                                                    className={selectedImage.detection_done ? 'border-accent-success/60 text-accent-success' : undefined}
                                                    onClick={handleToggleDetectionDone}
                                                    title="Shortcut: S"
                                                >
                                                    {selectedImage.detection_done ? 'Mark pending' : 'Mark done (S)'}
                                                </Button>
                                            )}
                                            {selectedImage && (
                                                <Button
                                                    size="sm"
                                                    variant="secondary"
                                                    className="border-accent-error/60 text-accent-error hover:text-white"
                                                    onClick={() => handleDeleteImage(selectedImage.id)}
                                                >
                                                    Delete image
                                                </Button>
                                            )}
                                            <Button
                                                size="sm"
                                                variant="secondary"
                                                onClick={goToPreviousDetectionImage}
                                                disabled={!canGoPrevDetection}
                                            >
                                                Previous
                                            </Button>
                                            <Button
                                                size="sm"
                                                variant="secondary"
                                                onClick={goToNextDetectionImage}
                                                disabled={!canGoNextDetection}
                                            >
                                                Next
                                            </Button>
                                        </div>
                                    </div>

                                    {selectedImage && (
                                        <div className="rounded-xl border border-graphite-700/70 bg-graphite-900/70 p-3">
                                            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                                                <div className="space-y-1">
                                                    <div className="text-[10px] uppercase tracking-[0.3em] text-secondary">File name</div>
                                                    <div className="text-xs text-bone truncate" title={selectedImage.path}>
                                                        {getFileNameFromPath(selectedImage.path)}
                                                    </div>
                                                </div>
                                                <div className="space-y-1">
                                                    <div className="text-[10px] uppercase tracking-[0.3em] text-secondary">Dimensions</div>
                                                    <div className="text-xs text-bone tabular-nums">
                                                        {selectedImageMetadata?.width && selectedImageMetadata?.height
                                                            ? `${selectedImageMetadata.width} x ${selectedImageMetadata.height}px`
                                                            : 'Loading...'}
                                                    </div>
                                                </div>
                                                <div className="space-y-1">
                                                    <div className="text-[10px] uppercase tracking-[0.3em] text-secondary">Resolution</div>
                                                    <div className="text-xs text-bone">
                                                        {selectedImageMetadata?.megapixels
                                                            ? `${selectedImageMetadata.megapixels.toFixed(2)} MP${selectedImageMetadata.width && selectedImageMetadata.height ? ` (${getResolutionTier(selectedImageMetadata.width, selectedImageMetadata.height)})` : ''}`
                                                            : 'Loading...'}
                                                    </div>
                                                </div>
                                                <div className="space-y-1">
                                                    <div className="text-[10px] uppercase tracking-[0.3em] text-secondary">Orientation</div>
                                                    <div className="text-xs text-bone">
                                                        {getOrientationLabel(selectedImageMetadata?.orientation)}
                                                    </div>
                                                </div>
                                                <div className="space-y-1">
                                                    <div className="text-[10px] uppercase tracking-[0.3em] text-secondary">Image size</div>
                                                    <div className="text-xs text-bone tabular-nums">
                                                        {typeof selectedImageMetadata?.fileSizeBytes === 'undefined'
                                                            ? 'Loading...'
                                                            : formatFileSize(selectedImageMetadata?.fileSizeBytes)}
                                                    </div>
                                                </div>
                                                <div className="space-y-1">
                                                    <div className="text-[10px] uppercase tracking-[0.3em] text-secondary">File id</div>
                                                    <div className="text-xs text-bone truncate" title={selectedImage.id}>
                                                        {selectedImage.id}
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    )}

                                    <div
                                        className="relative flex min-h-[560px] items-center justify-center border-2 border-graphite-700 bg-graphite-900 p-3"
                                        onMouseDown={handleMouseDown}
                                        onMouseMove={handleMouseMove}
                                        onMouseUp={handleMouseUp}
                                        onMouseLeave={() => {
                                            if (!isDrawing) return;
                                            setIsDrawing(false);
                                            setDraftBox(null);
                                        }}
                                    >
                                        {selectedImage ? (
                                            <div className="relative inline-block max-w-full">
                                                <img
                                                    ref={imageRef}
                                                    src={getImageSrc(selectedImage)}
                                                    alt={selectedImage.id}
                                                    className="max-h-[540px] max-w-full select-none object-contain"
                                                    draggable={false}
                                                    onLoad={(event) => {
                                                        handleSelectedImageLoad(event);
                                                        drawBoxes(selectedImage.boxes, draftBox);
                                                    }}
                                                />
                                                <canvas ref={canvasRef} className="absolute left-0 top-0 pointer-events-none" />
                                            </div>
                                        ) : (
                                            <div className="text-secondary text-sm">Select an image to start labeling.</div>
                                        )}
                                    </div>

                                    {selectedImage && (
                                        <div className="space-y-2">
                                            <div className="flex items-center justify-between text-xs uppercase tracking-[0.3em] text-haze">
                                                <span>Boxes</span>
                                                <span className="text-secondary">
                                                    {selectedImage.id} · {selectedImage.detection_done ? 'Done' : 'Pending'}
                                                </span>
                                            </div>
                                            {selectedImage.boxes.map((box, index) => (
                                                <div
                                                    key={`${selectedImage.id}-${index}`}
                                                    className={clsx(
                                                        'flex items-center justify-between border-2 px-2 py-1 text-xs',
                                                        editingBoxIndex === index ? 'border-accent-primary' : 'border-graphite-700'
                                                    )}
                                                >
                                                    <span>{classMap.get(box.class_id) || `Class ${box.class_id}`}</span>
                                                    <div className="flex items-center gap-3">
                                                        <button onClick={() => handleStartEditBox(index)} className="text-accent-primary">Edit</button>
                                                        <button onClick={() => handleRemoveBox(index)} className="text-error">Remove</button>
                                                    </div>
                                                </div>
                                            ))}
                                            {selectedImage.boxes.length === 0 && (
                                                <div className="text-xs text-secondary">No boxes yet.</div>
                                            )}
                                            {editingBoxDraft && (
                                                <div className="mt-3 rounded-xl border border-accent-primary/40 bg-graphite-900 p-3 space-y-3">
                                                    <div className="text-xs uppercase tracking-[0.3em] text-secondary">
                                                        Edit box #{(editingBoxIndex ?? 0) + 1}
                                                    </div>
                                                    <div className="grid gap-2 sm:grid-cols-2">
                                                        <Select
                                                            label="Class"
                                                            options={classes.map((cls) => ({
                                                                value: String(cls.id),
                                                                label: cls.name
                                                            }))}
                                                            value={String(editingBoxDraft.class_id)}
                                                            onChange={(event) => handleEditBoxFieldChange('class_id', event.target.value)}
                                                        />
                                                        <Input
                                                            label="X"
                                                            type="number"
                                                            min={0}
                                                            max={1}
                                                            step="0.001"
                                                            value={editingBoxDraft.x}
                                                            onChange={(event) => handleEditBoxFieldChange('x', event.target.value)}
                                                        />
                                                        <Input
                                                            label="Y"
                                                            type="number"
                                                            min={0}
                                                            max={1}
                                                            step="0.001"
                                                            value={editingBoxDraft.y}
                                                            onChange={(event) => handleEditBoxFieldChange('y', event.target.value)}
                                                        />
                                                        <Input
                                                            label="Width"
                                                            type="number"
                                                            min={0}
                                                            max={1}
                                                            step="0.001"
                                                            value={editingBoxDraft.w}
                                                            onChange={(event) => handleEditBoxFieldChange('w', event.target.value)}
                                                        />
                                                        <Input
                                                            label="Height"
                                                            type="number"
                                                            min={0}
                                                            max={1}
                                                            step="0.001"
                                                            value={editingBoxDraft.h}
                                                            onChange={(event) => handleEditBoxFieldChange('h', event.target.value)}
                                                        />
                                                    </div>
                                                    <div className="flex items-center justify-end gap-2">
                                                        <Button size="sm" variant="secondary" onClick={handleCancelEditBox}>
                                                            Cancel
                                                        </Button>
                                                        <Button size="sm" variant="primary" onClick={handleSaveEditedBox}>
                                                            Save box
                                                        </Button>
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    )}
                                </div>
                            </div>
                        </Card>
                    )}
                    {!dataset && (
                        <Card className="space-y-3">
                            <div className="text-secondary text-sm">No dataset metadata available for this project.</div>
                            <div className="flex gap-2">
                                <Button variant="secondary" onClick={fetchDataset}>
                                    Retry
                                </Button>
                                <Button variant="ghost" onClick={() => navigate('/projects')}>
                                    Back to Projects
                                </Button>
                            </div>
                        </Card>
                    )}
                </>
            )}
        </div>
    );
};
