import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { Select } from '../components/ui/Select';
import { ENGINE_HTTP_URL } from '../lib/engine';

interface Project {
    id: string;
    name: string;
    description: string;
    analysis_type?: 'classification' | 'detection';
    created_at: string;
    updated_at: string;
    datasets: string[];
}

const API_BASE_URL = ENGINE_HTTP_URL;

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
    const [form, setForm] = useState({ name: '', description: '', analysis_type: 'classification' });
    const [formError, setFormError] = useState<string | null>(null);
    const [creating, setCreating] = useState(false);

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
        } catch (err) {
            setFormError(err instanceof Error ? err.message : 'Failed to create project.');
        } finally {
            setCreating(false);
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
                        <Button variant="secondary" onClick={loadProjects} disabled={loading}>
                            {loading ? 'Refreshing...' : 'Refresh'}
                        </Button>
                    </div>
                </div>
                <div className="relative h-[2px] bg-graphite-700">
                    <span className="absolute left-0 -top-[3px] h-[6px] w-24 bg-accent-primary" />
                </div>
            </header>

            <div className="relative grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(280px,0.7fr)]">
                <div className="order-2 lg:order-1 space-y-6">
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
                                <div className="text-secondary">No projects yet. Create the first one on the right.</div>
                            </Card>
                        )}

                        {!loading && !error && projects.map((project) => (
                            <Card key={project.id} hoverEffect className="flex flex-col gap-4">
                                <div className="flex items-start justify-between">
                                    <div>
                                        <h3 className="font-semibold text-lg">{project.name}</h3>
                                        <div className="text-xs text-secondary font-mono">{project.id}</div>
                                    </div>
                                    <Button
                                        size="sm"
                                        variant="secondary"
                                        onClick={() => navigate(`/projects/${project.id}/dataset`)}
                                    >
                                        Open
                                    </Button>
                                </div>
                                <p className="text-secondary text-sm">
                                    {project.description || 'No description.'}
                                </p>
                                <div className="text-xs text-secondary">
                                    Created at {formatDate(project.created_at)}
                                </div>
                                <div className="text-xs text-secondary">
                                    Datasets: {project.datasets?.length || 0}
                                </div>
                                <div className="text-xs text-secondary uppercase tracking-[0.3em]">
                                    {project.analysis_type === 'detection' ? 'Detection' : 'Classification'}
                                </div>
                            </Card>
                        ))}
                    </div>
                </div>

                <div className="order-1 lg:order-2 lg:sticky lg:top-6 h-fit">
                    <Card className="relative">
                        <div className="absolute -left-6 top-10 origin-left -rotate-90 text-[10px] uppercase tracking-[0.6em] text-haze">
                            New Project
                        </div>
                        <h3 className="text-lg font-semibold mb-4">New Project</h3>
                        <div className="grid gap-3">
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
                            <Button variant="primary" onClick={handleCreate} disabled={creating}>
                                {creating ? 'Creating...' : 'Create project'}
                            </Button>
                        </div>
                    </Card>
                </div>
            </div>
        </div>
    );
};
