import { useEffect, useMemo, useState } from 'react';
import clsx from 'clsx';
import {
    AlertCircle,
    Bot,
    Brain,
    CheckCircle2,
    Cloud,
    Globe2,
    KeyRound,
    MessageCircleQuestion,
    Save,
    Sparkles,
    Zap
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { Select } from '../components/ui/Select';
import { ENGINE_HTTP_URL } from '../lib/engine';

type ProviderId = 'openrouter' | 'openai' | 'groq' | 'google';
type UseCaseId = 'preclassification' | 'chatbot' | 'image_validation';

interface ModelCatalog {
    id: string;
    label: string;
}

interface ProviderCatalog {
    id: ProviderId;
    name: string;
    description: string;
    default_base_url: string;
    models: ModelCatalog[];
}

interface UseCaseCatalog {
    id: UseCaseId;
    label: string;
    description: string;
}

interface ProviderSettings {
    enabled: boolean;
    api_key: string;
    base_url: string;
}

interface UseCaseAssignment {
    provider: ProviderId;
    model: string;
}

interface LlmSettings {
    active_provider: ProviderId;
    providers: Record<string, ProviderSettings>;
    assignments: Record<string, UseCaseAssignment>;
}

interface LlmSettingsResponse {
    settings: LlmSettings;
    catalog: {
        providers: ProviderCatalog[];
        use_cases: UseCaseCatalog[];
    };
}

const providerIconMap: Record<ProviderId, LucideIcon> = {
    openrouter: Globe2,
    openai: Sparkles,
    groq: Zap,
    google: Cloud
};

const useCaseIconMap: Record<UseCaseId, LucideIcon> = {
    preclassification: Brain,
    chatbot: MessageCircleQuestion,
    image_validation: Bot
};

const findProvider = (providers: ProviderCatalog[], providerId: string) =>
    providers.find((provider) => provider.id === providerId as ProviderId) || providers[0];

const firstProviderModel = (providers: ProviderCatalog[], providerId: string) => {
    const provider = findProvider(providers, providerId);
    return provider?.models?.[0]?.id || '';
};

const normalizeForActiveProvider = (rawSettings: LlmSettings, providers: ProviderCatalog[], useCases: UseCaseCatalog[]): LlmSettings => {
    const activeProvider = findProvider(providers, rawSettings.active_provider)?.id || providers[0]?.id;
    if (!activeProvider) return rawSettings;

    const assignments: Record<string, UseCaseAssignment> = { ...rawSettings.assignments };
    useCases.forEach((useCase) => {
        const current = assignments[useCase.id];
        const currentModel = current?.model || '';
        const providerModels = findProvider(providers, activeProvider)?.models || [];
        const hasModel = providerModels.some((model) => model.id === currentModel);
        assignments[useCase.id] = {
            provider: activeProvider,
            model: hasModel ? currentModel : firstProviderModel(providers, activeProvider)
        };
    });

    return {
        ...rawSettings,
        active_provider: activeProvider,
        assignments
    };
};

export const Settings = () => {
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [saveMessage, setSaveMessage] = useState<string | null>(null);
    const [providers, setProviders] = useState<ProviderCatalog[]>([]);
    const [useCases, setUseCases] = useState<UseCaseCatalog[]>([]);
    const [settings, setSettings] = useState<LlmSettings | null>(null);

    const activeProvider = useMemo(() => {
        if (!settings) return null;
        return findProvider(providers, settings.active_provider) || null;
    }, [settings, providers]);

    const loadSettings = async () => {
        setLoading(true);
        setError(null);
        setSaveMessage(null);
        try {
            const res = await fetch(`${ENGINE_HTTP_URL}/settings/llm`);
            if (!res.ok) {
                throw new Error('Could not load LLM settings.');
            }
            const data: LlmSettingsResponse = await res.json();
            setProviders(data.catalog.providers || []);
            setUseCases(data.catalog.use_cases || []);
            setSettings(normalizeForActiveProvider(data.settings, data.catalog.providers || [], data.catalog.use_cases || []));
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load settings.');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadSettings();
    }, []);

    const handleActiveProviderChange = (providerId: ProviderId) => {
        if (!settings) return;
        const next: LlmSettings = {
            ...settings,
            active_provider: providerId,
            assignments: { ...settings.assignments }
        };
        useCases.forEach((useCase) => {
            const useCaseId = useCase.id;
            const current = next.assignments[useCaseId];
            const models = findProvider(providers, providerId)?.models || [];
            const keepCurrent = current && models.some((model) => model.id === current.model);
            next.assignments[useCaseId] = {
                provider: providerId,
                model: keepCurrent ? current.model : firstProviderModel(providers, providerId)
            };
        });
        setSettings(next);
        setSaveMessage(null);
    };

    const handleUseCaseModelChange = (useCaseId: UseCaseId, modelId: string) => {
        if (!settings) return;
        setSettings({
            ...settings,
            assignments: {
                ...settings.assignments,
                [useCaseId]: {
                    provider: settings.active_provider,
                    model: modelId
                }
            }
        });
        setSaveMessage(null);
    };

    const handleProviderFieldChange = (providerId: ProviderId, field: keyof ProviderSettings, value: string | boolean) => {
        if (!settings) return;
        const current = settings.providers[providerId] || { enabled: true, api_key: '', base_url: '' };
        setSettings({
            ...settings,
            providers: {
                ...settings.providers,
                [providerId]: {
                    ...current,
                    [field]: value
                }
            }
        });
        setSaveMessage(null);
    };

    const handleSave = async () => {
        if (!settings) return;
        setSaving(true);
        setError(null);
        setSaveMessage(null);
        try {
            const res = await fetch(`${ENGINE_HTTP_URL}/settings/llm`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(settings)
            });
            if (!res.ok) {
                throw new Error('Could not save settings.');
            }
            const data: LlmSettingsResponse = await res.json();
            setProviders(data.catalog.providers || []);
            setUseCases(data.catalog.use_cases || []);
            setSettings(normalizeForActiveProvider(data.settings, data.catalog.providers || [], data.catalog.use_cases || []));
            setSaveMessage('Settings saved successfully.');
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to save settings.');
        } finally {
            setSaving(false);
        }
    };

    const activeProviderModels = activeProvider?.models || [];

    return (
        <div className="animate-fade-in space-y-6">
            <header className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
                <div>
                    <div className="text-xs uppercase tracking-[0.6em] text-haze">Settings</div>
                    <h2 className="text-3xl font-semibold">Multimodal LLM Configuration</h2>
                    <p className="text-secondary text-sm mt-2">
                        Configure multimodal providers, models, and use cases for the app.
                    </p>
                </div>
                <div className="flex items-center gap-2">
                    <Button variant="secondary" onClick={loadSettings} disabled={loading || saving}>
                        {loading ? 'Loading...' : 'Refresh'}
                    </Button>
                    <Button variant="primary" onClick={handleSave} disabled={!settings || saving || loading}>
                        <Save size={14} />
                        {saving ? 'Saving...' : 'Save settings'}
                    </Button>
                </div>
            </header>

            {error && (
                <Card className="border border-accent-error/50">
                        <div className="flex items-center gap-2 text-error text-sm">
                        <AlertCircle size={16} />
                        {error}
                    </div>
                </Card>
            )}

            {saveMessage && (
                <Card className="border border-accent-success/40">
                    <div className="flex items-center gap-2 text-success text-sm">
                        <CheckCircle2 size={16} />
                        {saveMessage}
                    </div>
                </Card>
            )}

            {loading || !settings ? (
                <Card>
                    <div className="text-secondary">Loading LLM settings...</div>
                </Card>
            ) : (
                <>
                    <Card className="space-y-4">
                        <div>
                            <h3 className="text-lg font-semibold">Active Provider</h3>
                            <p className="text-secondary text-sm">
                                Choose the main multimodal provider for the system.
                            </p>
                        </div>
                        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                            {providers.map((provider) => {
                                const Icon = providerIconMap[provider.id] || Globe2;
                                const isActive = settings.active_provider === provider.id;
                                return (
                                    <button
                                        key={provider.id}
                                        type="button"
                                        onClick={() => handleActiveProviderChange(provider.id)}
                                        className={clsx(
                                            'rounded-2xl border px-4 py-4 text-left transition-all',
                                            isActive
                                                ? 'border-accent-primary bg-accent-primary/10 shadow-[0_0_0_1px_rgba(47,184,255,0.4)]'
                                                : 'border-graphite-700/70 bg-graphite-900 hover:border-accent-secondary'
                                        )}
                                    >
                                        <div className="flex items-center justify-between gap-2">
                                            <div className="rounded-xl border border-graphite-700/70 bg-graphite-950/60 p-2">
                                                <Icon size={16} className={isActive ? 'text-accent-primary' : 'text-haze'} />
                                            </div>
                                            {isActive && (
                                                <span className="text-[10px] uppercase tracking-[0.3em] text-accent-primary">
                                                    Active
                                                </span>
                                            )}
                                        </div>
                                        <div className="mt-3 text-sm font-semibold">{provider.name}</div>
                                        <div className="mt-1 text-xs text-secondary">{provider.description}</div>
                                    </button>
                                );
                            })}
                        </div>
                    </Card>

                    <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
                        <Card className="space-y-4">
                            <div className="flex items-center gap-2">
                                <Brain size={16} className="text-accent-primary" />
                                <h3 className="text-lg font-semibold">Model Assignment by Use Case</h3>
                            </div>
                            <p className="text-secondary text-sm">
                                Define which multimodal model will be used for each flow under the active provider.
                            </p>
                            <div className="rounded-xl border border-graphite-700/70 bg-graphite-900 px-3 py-2 text-xs text-secondary">
                                Active provider for all 3 flows: <span className="text-bone font-semibold">{activeProvider?.name || 'N/A'}</span>
                            </div>

                            <div className="space-y-3">
                                {useCases.map((useCase) => {
                                    const Icon = useCaseIconMap[useCase.id] || Bot;
                                    const currentModel = settings.assignments[useCase.id]?.model || firstProviderModel(providers, settings.active_provider);
                                    return (
                                        <div key={useCase.id} className="rounded-xl border border-graphite-700/70 bg-graphite-900 p-3">
                                            <div className="flex items-start gap-3">
                                                <div className="rounded-lg border border-graphite-700/70 bg-graphite-950/70 p-2">
                                                    <Icon size={14} className="text-accent-primary" />
                                                </div>
                                                <div className="flex-1 space-y-2">
                                                    <div>
                                                        <div className="text-sm font-semibold">{useCase.label}</div>
                                                        <div className="text-xs text-secondary mt-1">{useCase.description}</div>
                                                    </div>
                                                    <Select
                                                        label="Multimodal model"
                                                        options={activeProviderModels.map((model) => ({
                                                            value: model.id,
                                                            label: model.label
                                                        }))}
                                                        value={currentModel}
                                                        onChange={(event) => handleUseCaseModelChange(useCase.id, event.target.value)}
                                                    />
                                                </div>
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </Card>

                        <Card className="space-y-4">
                            <div className="flex items-center gap-2">
                                <KeyRound size={16} className="text-accent-primary" />
                                <h3 className="text-lg font-semibold">Provider Credentials</h3>
                            </div>
                            <p className="text-secondary text-sm">
                                Provider keys and endpoints for future multimodal features.
                            </p>

                            <div className="space-y-3">
                                {providers.map((provider) => {
                                    const Icon = providerIconMap[provider.id] || Globe2;
                                    const providerSettings = settings.providers[provider.id] || {
                                        enabled: true,
                                        api_key: '',
                                        base_url: provider.default_base_url
                                    };
                                    return (
                                        <div key={`credentials-${provider.id}`} className="rounded-xl border border-graphite-700/70 bg-graphite-900 p-3">
                                            <div className="flex items-center justify-between gap-2">
                                                <div className="flex items-center gap-2">
                                                    <div className="rounded-lg border border-graphite-700/70 bg-graphite-950/60 p-2">
                                                        <Icon size={14} className="text-haze" />
                                                    </div>
                                                    <div>
                                                        <div className="text-sm font-semibold">{provider.name}</div>
                                                        <div className="text-[11px] text-secondary">Models: {provider.models.length}</div>
                                                    </div>
                                                </div>
                                                <label className="flex items-center gap-2 text-xs text-secondary">
                                                    <input
                                                        type="checkbox"
                                                        checked={providerSettings.enabled}
                                                        onChange={(event) =>
                                                            handleProviderFieldChange(provider.id, 'enabled', event.target.checked)
                                                        }
                                                    />
                                                    Enabled
                                                </label>
                                            </div>
                                            <div className="mt-3 grid gap-2">
                                                <Input
                                                    label="API Key"
                                                    type="password"
                                                    placeholder={`Key for ${provider.name}`}
                                                    value={providerSettings.api_key}
                                                    onChange={(event) =>
                                                        handleProviderFieldChange(provider.id, 'api_key', event.target.value)
                                                    }
                                                />
                                                <Input
                                                    label="Base URL"
                                                    placeholder={provider.default_base_url}
                                                    value={providerSettings.base_url}
                                                    onChange={(event) =>
                                                        handleProviderFieldChange(provider.id, 'base_url', event.target.value)
                                                    }
                                                />
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </Card>
                    </div>
                </>
            )}
        </div>
    );
};
