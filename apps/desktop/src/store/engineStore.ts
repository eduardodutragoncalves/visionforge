import { create } from 'zustand';

interface GpuState {
    index: number;
    name: string;
    memoryUsed: number;
    memoryTotal: number;
    utilization: number;
    temperature: number;
}

interface GpuTimelineEntry {
    timestamp: number;
    gpus: GpuState[];
}

interface EngineState {
    isConnected: boolean;
    status: 'offline' | 'idle' | 'busy';
    gpu: GpuState;
    gpus: GpuState[];
    gpuTimeline: GpuTimelineEntry[];
    dashboardReady: boolean;

    setConnected: (connected: boolean) => void;
    setGpuStats: (gpus: GpuState[], timestamp: number) => void;
    setEngineStatus: (status: EngineState['status']) => void;
    setDashboardReady: (ready: boolean) => void;
}

export const useEngineStore = create<EngineState>((set) => ({
    isConnected: false,
    status: 'offline',
    gpu: {
        index: 0,
        name: 'N/A',
        memoryUsed: 0,
        memoryTotal: 0,
        utilization: 0,
        temperature: 0
    },
    gpus: [],
    gpuTimeline: [],
    dashboardReady: false,

    setConnected: (connected) =>
        set({
            isConnected: connected,
            status: connected ? 'idle' : 'offline'
        }),
    setEngineStatus: (status) => set({ status }),
    setDashboardReady: (ready) => set({ dashboardReady: ready }),
    setGpuStats: (gpus, timestamp) =>
        set((state) => {
            const entry: GpuTimelineEntry = { timestamp, gpus };
            const timeline = [...state.gpuTimeline, entry].slice(-300);
            const primary = gpus[0] ?? state.gpu;
            return { gpus, gpu: primary, gpuTimeline: timeline };
        })
}));
