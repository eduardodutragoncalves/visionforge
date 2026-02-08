import { create } from 'zustand';

export interface MetricPoint {
    epoch: number;
    loss: number;
    accuracy: number;
    lr: number;
    total_epochs?: number;
}

export interface LogEntry {
    timestamp: string;
    level: string;
    message: string;
}

interface TrainingState {
    runs: Record<string, RunState>;
    activeRunId: string | null;

    startRun: (runId: string, totalEpochs: number, projectId?: string, config?: Record<string, any>) => void;
    addMetric: (runId: string, metric: MetricPoint) => void;
    addLog: (runId: string, log: LogEntry) => void;
    setStatus: (runId: string, status: RunState['status']) => void;
    setActiveRun: (runId: string | null) => void;
    resetRun: (runId: string) => void;
    resetAll: () => void;
}

interface RunState {
    runId: string;
    projectId?: string;
    status: 'idle' | 'starting' | 'training' | 'completed' | 'failed' | 'stopped';
    currentEpoch: number;
    totalEpochs: number;
    metrics: MetricPoint[];
    logs: LogEntry[];
    startedAt?: number;
    config?: Record<string, any>;
}

const createRunState = (runId: string, totalEpochs: number, projectId?: string, config?: Record<string, any>): RunState => ({
    runId,
    projectId,
    status: 'starting',
    totalEpochs,
    metrics: [],
    logs: [],
    currentEpoch: 0,
    startedAt: Date.now(),
    config
});

export const useTrainingStore = create<TrainingState>((set) => ({
    runs: {},
    activeRunId: null,

    startRun: (runId, totalEpochs, projectId, config) => set((state) => {
        const existing = state.runs[runId];
        const next = existing
            ? {
                ...existing,
                status: existing.status === 'idle' ? 'starting' : existing.status,
                totalEpochs: totalEpochs || existing.totalEpochs,
                projectId: projectId ?? existing.projectId,
                config: config ?? existing.config
            }
            : createRunState(runId, totalEpochs, projectId, config);
        return {
            activeRunId: runId,
            runs: {
                ...state.runs,
                [runId]: next
            }
        };
    }),

    addMetric: (runId, metric) => set((state) => {
        const run = state.runs[runId] ?? createRunState(runId, metric.total_epochs || 0);
        return {
            runs: {
                ...state.runs,
                [runId]: {
                    ...run,
                    metrics: [...run.metrics, metric],
                    currentEpoch: metric.epoch,
                    totalEpochs: metric.total_epochs || run.totalEpochs,
                    status: run.status === 'starting' ? 'training' : run.status
                }
            }
        };
    }),

    addLog: (runId, log) => set((state) => {
        const run = state.runs[runId] ?? createRunState(runId, 0);
        return {
            runs: {
                ...state.runs,
                [runId]: {
                    ...run,
                    logs: [...run.logs.slice(-500), log]
                }
            }
        };
    }),

    setStatus: (runId, status) => set((state) => {
        const run = state.runs[runId] ?? createRunState(runId, 0);
        return {
            runs: {
                ...state.runs,
                [runId]: {
                    ...run,
                    status
                }
            }
        };
    }),

    setActiveRun: (runId) => set({ activeRunId: runId }),

    resetRun: (runId) => set((state) => {
        if (!state.runs[runId]) return state;
        const next = { ...state.runs };
        delete next[runId];
        return {
            runs: next,
            activeRunId: state.activeRunId === runId ? null : state.activeRunId
        };
    }),

    resetAll: () => set({ activeRunId: null, runs: {} })
}));
