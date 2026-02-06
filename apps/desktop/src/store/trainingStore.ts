import { create } from 'zustand';

export interface MetricPoint {
    epoch: number;
    loss: number;
    accuracy: number;
    lr: number;
}

export interface LogEntry {
    timestamp: string;
    level: string;
    message: string;
}

interface TrainingState {
    activeRunId: string | null;
    status: 'idle' | 'starting' | 'training' | 'completed' | 'failed' | 'stopped';
    currentEpoch: number;
    totalEpochs: number;
    metrics: MetricPoint[];
    logs: LogEntry[];

    startRun: (runId: string, totalEpochs: number) => void;
    addMetric: (metric: MetricPoint) => void;
    addLog: (log: LogEntry) => void;
    setStatus: (status: TrainingState['status']) => void;
    reset: () => void;
}

export const useTrainingStore = create<TrainingState>((set) => ({
    activeRunId: null,
    status: 'idle',
    currentEpoch: 0,
    totalEpochs: 0,
    metrics: [],
    logs: [],

    startRun: (runId, totalEpochs) => set({
        activeRunId: runId,
        status: 'starting',
        totalEpochs,
        metrics: [],
        logs: [],
        currentEpoch: 0
    }),

    addMetric: (metric) => set((state) => ({
        metrics: [...state.metrics, metric],
        currentEpoch: metric.epoch
    })),

    addLog: (log) => set((state) => ({
        logs: [...state.logs.slice(-500), log] // Keep last 500 logs
    })),

    setStatus: (status) => set({ status }),
    reset: () => set({ activeRunId: null, status: 'idle', metrics: [], logs: [] })
}));
