import { create } from 'zustand';

interface GpuState {
    name: string;
    memoryUsed: number;
    memoryTotal: number;
    utilization: number;
    temperature: number;
}

interface EngineState {
    isConnected: boolean;
    status: 'offline' | 'idle' | 'busy';
    gpu: GpuState;

    setConnected: (connected: boolean) => void;
    updateGpu: (stats: Partial<GpuState>) => void;
    setEngineStatus: (status: EngineState['status']) => void;
}

export const useEngineStore = create<EngineState>((set) => ({
    isConnected: false,
    status: 'offline',
    gpu: {
        name: 'N/A',
        memoryUsed: 0,
        memoryTotal: 0,
        utilization: 0,
        temperature: 0
    },

    setConnected: (connected) => set({ isConnected: connected, status: connected ? 'idle' : 'offline' }),
    setEngineStatus: (status) => set({ status }),
    updateGpu: (stats) => set((state) => ({ gpu: { ...state.gpu, ...stats } }))
}));
