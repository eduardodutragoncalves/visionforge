import { useMemo, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import clsx from 'clsx';
import { Boxes, ListFilter, Shapes } from 'lucide-react';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { Select } from '../components/ui/Select';
import { ENGINE_HTTP_URL, ENGINE_WS_URL } from '../lib/engine';

interface Project {
    id: string;
    name: string;
    description: string;
    analysis_type?: 'classification' | 'detection';
    created_at: string;
    updated_at: string;
    datasets: string[];
    images_total?: number;
    labeled_images?: number;
    labels_total?: number;
    classes_total?: number;
    sources_total?: number;
    runs_total?: number;
    last_run_at?: string | null;
}

interface DeleteState {
    projectId: string;
    projectName: string;
    status: string;
    progress: number;
    total?: number;
    deleted?: number;
}

const API_BASE_URL = ENGINE_HTTP_URL;
type ProjectTypeFilter = 'all' | 'classification' | 'detection';

const normalizeAnalysisType = (type: Project['analysis_type']): 'classification' | 'detection' =>
    type === 'detection' ? 'detection' : 'classification';

const formatDate = (iso: string) => {
    const parsed = new Date(iso);
    if (Number.isNaN(parsed.getTime())) {
        return iso;
    }
    return parsed.toLocaleString();
};

export const Projects = () => {
    const navigate = useNavigate();
    const [projects, setProjects] = useState<Project[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
    const [form, setForm] = useState({ name: '', description: '', analysis_type: 'classification' });
    const [formError, setFormError] = useState<string | null>(null);
    const [creating, setCreating] = useState(false);
    const [deleteState, setDeleteState] = useState<DeleteState | null>(null);
    const [deleteError, setDeleteError] = useState<string | null>(null);
    const [typeFilter, setTypeFilter] = useState<ProjectTypeFilter>('all');
    const deleteStateRef = useRef<DeleteState | null>(null);
    const filteredProjects = useMemo(() => {
        if (typeFilter === 'all') return projects;
        return projects.filter((project) => normalizeAnalysisType(project.analysis_type) === typeFilter);
    }, [projects, typeFilter]);
    const classificationCount = useMemo(
        () => projects.filter((project) => normalizeAnalysisType(project.analysis_type) === 'classification').length,
        [projects]
    );
    const detectionCount = useMemo(
        () => projects.filter((project) => normalizeAnalysisType(project.analysis_type) === 'detection').length,
        [projects]
    );

    const loadProjects = async () => {
        setLoading(true);
        setError(null);
        try {
            const res = await fetch(`${API_BASE_URL}/projects`);
            if (!res.ok) {
                throw new Error('Failed to load projects.');
            }
            const data = await res.json();
            setProjects(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load projects.');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadProjects();
    }, []);

    useEffect(() => {
        deleteStateRef.current = deleteState;
    }, [deleteState]);

    useEffect(() => {
        const ws = new WebSocket(ENGINE_WS_URL);
        ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                if (msg.event !== 'project/delete_progress') return;
                const current = deleteStateRef.current;
                if (!current || msg.data.project_id !== current.projectId) return;
                setDeleteState((prev) => {
                    if (!prev) return prev;
                    return {
                        ...prev,
                        status: msg.data.status || prev.status,
                        progress: Number(msg.data.progress ?? prev.progress),
                        total: msg.data.total ?? prev.total,
                        deleted: msg.data.deleted ?? prev.deleted
                    };
                });
                if (msg.data.status === 'completed') {
                    setProjects((prev) => prev.filter((project) => project.id !== current.projectId));
                }
            } catch {
                // ignore
            }
        };
        return () => ws.close();
    }, []);

    useEffect(() => {
        if (!deleteState || deleteState.status === 'completed' || deleteState.status === 'error') return;
        let canceled = false;
        const interval = setInterval(async () => {
            try {
                const res = await fetch(`${API_BASE_URL}/projects/${deleteState.projectId}`);
                if (res.status === 404 && !canceled) {
                    setDeleteState((prev) => (prev ? { ...prev, status: 'completed', progress: 100 } : prev));
                    setProjects((prev) => prev.filter((project) => project.id !== deleteState.projectId));
                }
            } catch {
                // ignore
            }
        }, 2000);
        return () => {
            canceled = true;
            clearInterval(interval);
        };
    }, [deleteState?.projectId, deleteState?.status]);

    const handleCreate = async () => {
        const trimmedName = form.name.trim();
        if (!trimmedName) {
            setFormError('Please enter a project name.');
            return;
        }

        setFormError(null);
        setCreating(true);
        try {
            const res = await fetch(`${API_BASE_URL}/projects`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: trimmedName,
                    description: form.description.trim(),
                    analysis_type: form.analysis_type
                })
            });

            if (!res.ok) {
                throw new Error('Failed to create project.');
            }

            const created = await res.json();
            setProjects((prev) => [created, ...prev]);
            setForm({ name: '', description: '', analysis_type: 'classification' });
            setIsCreateModalOpen(false);
        } catch (err) {
            setFormError(err instanceof Error ? err.message : 'Failed to create project.');
        } finally {
            setCreating(false);
        }
    };

    const handleDelete = async (project: Project) => {
        const confirmed = window.confirm(`Delete project "${project.name}" and all its data?`);
        if (!confirmed) return;
        setDeleteError(null);
        setDeleteState({
            projectId: project.id,
            projectName: project.name,
            status: 'starting',
            progress: 0
        });
        try {
            const res = await fetch(`${API_BASE_URL}/projects/${project.id}`, { method: 'DELETE' });
            if (!res.ok) {
                throw new Error('Failed to delete project.');
            }
        } catch (err) {
            setDeleteError(err instanceof Error ? err.message : 'Failed to delete project.');
            setDeleteState((prev) => (prev ? { ...prev, status: 'error' } : prev));
        }
    };

    return (
        <div className="animate-fade-in space-y-8">
            <header className="space-y-6">
                <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                    <div className="max-w-2xl">
                        <div className="text-xs uppercase tracking-[0.6em] text-haze">Workspace</div>
                        <h2 className="text-4xl md:text-5xl font-semibold tracking-tight">Projects</h2>
                        <p className="text-secondary text-sm mt-2">
                            Create, organize, and open projects with a direct flow to dataset, training, and testing.
                        </p>
                    </div>
                    <div className="flex items-center gap-3">
                        <Button
                            variant="primary"
                            size="lg"
                            className="min-w-[220px]"
                            onClick={() => {
                                setFormError(null);
                                setIsCreateModalOpen(true);
                            }}
                        >
                            New Project
                        </Button>
                        <Button variant="secondary" onClick={loadProjects} disabled={loading}>
                            {loading ? 'Refreshing...' : 'Refresh'}
                        </Button>
                    </div>
                </div>
                <div className="relative h-[2px] bg-graphite-700">
                    <span className="absolute left-0 -top-[3px] h-[6px] w-24 bg-accent-primary" />
                </div>
            </header>

            <div className="space-y-6">
                <div className="flex flex-wrap items-center gap-2">
                    <button
                        type="button"
                        className={clsx('tab-button inline-flex items-center gap-2', typeFilter === 'all' && 'tab-button-active')}
                        onClick={() => setTypeFilter('all')}
                    >
                        <ListFilter size={14} />
                        <span>All ({projects.length})</span>
                    </button>
                    <button
                        type="button"
                        className={clsx('tab-button inline-flex items-center gap-2', typeFilter === 'classification' && 'tab-button-active')}
                        onClick={() => setTypeFilter('classification')}
                    >
                        <Shapes size={14} />
                        <span>Classification ({classificationCount})</span>
                    </button>
                    <button
                        type="button"
                        className={clsx('tab-button inline-flex items-center gap-2', typeFilter === 'detection' && 'tab-button-active')}
                        onClick={() => setTypeFilter('detection')}
                    >
                        <Boxes size={14} />
                        <span>Object Detection ({detectionCount})</span>
                    </button>
                </div>
                <div className="stagger-children grid gap-6 sm:grid-cols-2">
                    {loading && (
                        <Card>
                            <div className="text-secondary">Loading projects...</div>
                        </Card>
                    )}

                    {!loading && error && (
                        <Card>
                            <div className="text-error">{error}</div>
                            <div className="text-secondary text-sm mt-2">Check if the engine is running.</div>
                        </Card>
                    )}

                    {!loading && !error && projects.length === 0 && (
                        <Card>
                            <div className="text-secondary">No projects yet. Create the first one with New Project.</div>
                        </Card>
                    )}

                    {!loading && !error && projects.length > 0 && filteredProjects.length === 0 && (
                        <Card>
                            <div className="text-secondary">No projects match this filter.</div>
                        </Card>
                    )}

                    {!loading && !error && filteredProjects.map((project) => (
                        <Card key={project.id} hoverEffect className="flex flex-col gap-4">
                            <div className="flex items-start justify-between">
                                <div>
                                    <h3 className="font-semibold text-lg">{project.name}</h3>
                                    <div className="text-xs text-secondary font-mono">{project.id}</div>
                                </div>
                                <div className="flex items-center gap-2">
                                    <Button
                                        size="sm"
                                        variant="secondary"
                                        onClick={() => navigate(`/projects/${project.id}/dataset`)}
                                    >
                                        Open
                                    </Button>
                                    <Button
                                        size="sm"
                                        variant="secondary"
                                        className="border-accent-error/60 text-accent-error hover:text-white"
                                        onClick={() => handleDelete(project)}
                                    >
                                        Delete
                                    </Button>
                                </div>
                            </div>
                            <p className="text-secondary text-sm">
                                {project.description || 'No description.'}
                            </p>
                            <div className="grid gap-2 text-xs text-secondary">
                                <div>Created at {formatDate(project.created_at)}</div>
                                {project.last_run_at && (
                                    <div>Last run {formatDate(project.last_run_at)}</div>
                                )}
                            </div>
                            <div className="grid gap-2 sm:grid-cols-2 text-xs">
                                <div className="stat-chip">
                                    <span>Images</span>
                                    <strong>{project.images_total ?? 0}</strong>
                                </div>
                                <div className="stat-chip">
                                    <span>Labeled</span>
                                    <strong>{project.labeled_images ?? 0}</strong>
                                </div>
                                <div className="stat-chip">
                                    <span>Labels</span>
                                    <strong>{project.labels_total ?? 0}</strong>
                                </div>
                                <div className="stat-chip">
                                    <span>Runs</span>
                                    <strong>{project.runs_total ?? 0}</strong>
                                </div>
                            </div>
                            <div className="flex flex-wrap gap-2 text-xs uppercase tracking-[0.3em] text-secondary">
                                <span>{project.analysis_type === 'detection' ? 'Detection' : 'Classification'}</span>
                                <span>Classes {project.classes_total ?? 0}</span>
                                <span>Sources {project.sources_total ?? 0}</span>
                            </div>
                        </Card>
                    ))}
                </div>
            </div>
            {isCreateModalOpen && (
                <div className="modal-backdrop" role="dialog" aria-modal="true">
                    <div className="modal-card max-w-[560px]">
                        <div className="modal-header">
                            <div>
                                <div className="text-xs uppercase tracking-[0.4em] text-haze">Workspace</div>
                                <div className="text-lg font-semibold">New Project</div>
                            </div>
                            <button
                                type="button"
                                className="window-button"
                                onClick={() => {
                                    setFormError(null);
                                    setIsCreateModalOpen(false);
                                }}
                                aria-label="Close"
                            >
                                ✕
                            </button>
                        </div>
                        <div className="mt-5 grid gap-3">
                            <Input
                                label="Name"
                                placeholder="Ex: Defect detection"
                                value={form.name}
                                onChange={(e) => setForm((prev) => ({ ...prev, name: e.target.value }))}
                            />
                            <Select
                                label="Analysis Type"
                                options={[
                                    { value: 'classification', label: 'Classification' },
                                    { value: 'detection', label: 'Object Detection' }
                                ]}
                                value={form.analysis_type}
                                onChange={(e) => setForm((prev) => ({ ...prev, analysis_type: e.target.value }))}
                            />
                            <Input
                                label="Description"
                                placeholder="Short project context"
                                value={form.description}
                                onChange={(e) => setForm((prev) => ({ ...prev, description: e.target.value }))}
                            />
                            {formError && (
                                <div className="text-xs text-error">{formError}</div>
                            )}
                            <div className="mt-2 flex items-center justify-end gap-2">
                                <Button
                                    variant="secondary"
                                    onClick={() => {
                                        setFormError(null);
                                        setIsCreateModalOpen(false);
                                    }}
                                >
                                    Cancel
                                </Button>
                                <Button variant="primary" onClick={handleCreate} disabled={creating}>
                                    {creating ? 'Creating...' : 'Create project'}
                                </Button>
                            </div>
                        </div>
                    </div>
                </div>
            )}
            {deleteState && (
                <div className="modal-backdrop" role="dialog" aria-modal="true">
                    <div className="modal-card">
                        <div className="modal-header">
                            <div>
                                <div className="text-xs uppercase tracking-[0.4em] text-haze">Deleting project</div>
                                <div className="text-lg font-semibold">{deleteState.projectName}</div>
                            </div>
                            {(deleteState.status === 'completed' || deleteState.status === 'error') && (
                                <button
                                    type="button"
                                    className="window-button"
                                    onClick={() => setDeleteState(null)}
                                    aria-label="Close"
                                >
                                    ✕
                                </button>
                            )}
                        </div>
                        <div className="mt-4 text-sm text-secondary">
                            {deleteState.status === 'completed'
                                ? 'Project deleted.'
                                : deleteState.status === 'error'
                                    ? 'Failed to delete project.'
                                    : 'Removing files and metadata...'}
                        </div>
                        <div className="mt-4 h-2 bg-graphite-700 rounded overflow-hidden">
                            <div
                                className="h-full bg-accent-error transition-all duration-300"
                                style={{ width: `${Math.min(deleteState.progress, 100)}%` }}
                            />
                        </div>
                        <div className="mt-3 text-xs text-secondary">
                            {deleteState.deleted ?? 0} / {deleteState.total ?? 0} items removed
                        </div>
                        {deleteError && (
                            <div className="mt-3 text-xs text-error">{deleteError}</div>
                        )}
                        {(deleteState.status === 'completed' || deleteState.status === 'error') && (
                            <div className="mt-4 flex justify-end">
                                <Button size="sm" variant="secondary" onClick={() => setDeleteState(null)}>
                                    Done
                                </Button>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};
