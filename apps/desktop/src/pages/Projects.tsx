import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';

interface Project {
    id: string;
    name: string;
    description: string;
    created_at: string;
    updated_at: string;
    datasets: string[];
}

const API_BASE_URL = 'http://localhost:8000';

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
    const [form, setForm] = useState({ name: '', description: '' });
    const [formError, setFormError] = useState<string | null>(null);
    const [creating, setCreating] = useState(false);

    const loadProjects = async () => {
        setLoading(true);
        setError(null);
        try {
            const res = await fetch(`${API_BASE_URL}/projects`);
            if (!res.ok) {
                throw new Error('Falha ao carregar projetos.');
            }
            const data = await res.json();
            setProjects(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Falha ao carregar projetos.');
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
            setFormError('Informe um nome para o projeto.');
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
                    description: form.description.trim()
                })
            });

            if (!res.ok) {
                throw new Error('Falha ao criar projeto.');
            }

            const created = await res.json();
            setProjects((prev) => [created, ...prev]);
            setForm({ name: '', description: '' });
        } catch (err) {
            setFormError(err instanceof Error ? err.message : 'Falha ao criar projeto.');
        } finally {
            setCreating(false);
        }
    };

    return (
        <div className="animate-fade-in">
            <header className="flex justify-between items-center mb-6">
                <div>
                    <h2 className="text-2xl font-bold">Projects</h2>
                    <p className="text-secondary">Crie, organize e abra seus projetos.</p>
                </div>
                <Button variant="secondary" onClick={loadProjects} disabled={loading}>
                    {loading ? 'Atualizando...' : 'Atualizar'}
                </Button>
            </header>

            <div style={{ display: 'grid', gap: 24, marginBottom: 24 }}>
                <Card>
                    <h3 className="text-lg font-bold mb-4">Novo Projeto</h3>
                    <div style={{ display: 'grid', gap: 12 }}>
                        <Input
                            label="Nome"
                            placeholder="Ex: Detecção de defeitos"
                            value={form.name}
                            onChange={(e) => setForm((prev) => ({ ...prev, name: e.target.value }))}
                        />
                        <Input
                            label="Descrição"
                            placeholder="Breve contexto do projeto"
                            value={form.description}
                            onChange={(e) => setForm((prev) => ({ ...prev, description: e.target.value }))}
                        />
                        {formError && (
                            <div className="text-xs text-error">{formError}</div>
                        )}
                        <Button variant="primary" onClick={handleCreate} disabled={creating}>
                            {creating ? 'Criando...' : 'Criar projeto'}
                        </Button>
                    </div>
                </Card>

                <div style={{ display: 'grid', gap: 24, gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))' }}>
                    {loading && (
                        <Card>
                            <div className="text-secondary">Carregando projetos...</div>
                        </Card>
                    )}

                    {!loading && error && (
                        <Card>
                            <div className="text-error">{error}</div>
                            <div className="text-secondary text-sm mt-2">Verifique se o engine está rodando.</div>
                        </Card>
                    )}

                    {!loading && !error && projects.length === 0 && (
                        <Card>
                            <div className="text-secondary">Nenhum projeto encontrado. Crie o primeiro acima.</div>
                        </Card>
                    )}

                    {!loading && !error && projects.map((project) => (
                        <Card key={project.id} hoverEffect>
                            <div className="flex justify-between items-start mb-4">
                                <div>
                                    <h3 className="font-bold text-lg">{project.name}</h3>
                                    <div className="text-xs text-secondary font-mono">{project.id}</div>
                                </div>
                                <Button
                                    size="sm"
                                    variant="secondary"
                                    onClick={() => navigate(`/projects/${project.id}/dataset`)}
                                >
                                    Abrir
                                </Button>
                            </div>
                            <p className="text-secondary text-sm mb-4">
                                {project.description || 'Sem descrição.'}
                            </p>
                            <div className="text-xs text-secondary">
                                Criado em {formatDate(project.created_at)}
                            </div>
                            <div className="text-xs text-secondary mt-1">
                                Datasets: {project.datasets?.length || 0}
                            </div>
                        </Card>
                    ))}
                </div>
            </div>
        </div>
    );
};
