import { useRef, useEffect } from 'react';
import { useEngineStore } from '../store/engineStore';
import { useTrainingStore } from '../store/trainingStore';
import { ENGINE_WS_URL } from '../lib/engine';

export const useEngineConnection = () => {
    const wsRef = useRef<WebSocket | null>(null);
    const { setConnected, setGpuStats } = useEngineStore();
    const { addMetric, addLog, setStatus, startRun } = useTrainingStore();

    useEffect(() => {
        const connect = () => {
            const ws = new WebSocket(ENGINE_WS_URL);
            wsRef.current = ws;

            ws.onopen = () => {
                setConnected(true);
                console.log("WS Connected");
            };

            ws.onclose = () => {
                setConnected(false);
                console.log("WS Disconnected, retrying in 3s...");
                setTimeout(connect, 3000);
            };

            ws.onmessage = (event) => {
                try {
                    const msg = JSON.parse(event.data);
                    if (msg.event === 'train/metrics') {
                        addMetric(msg.data);
                        setStatus('training');
                    } else if (msg.event === 'engine/gpu_stats') {
                        const gpus = (msg.data?.gpus || []).map((gpu: any) => ({
                            index: gpu.index ?? 0,
                            name: gpu.name ?? 'GPU',
                            utilization: gpu.utilization ?? 0,
                            memoryUsed: gpu.memory_used ?? gpu.memoryUsed ?? 0,
                            memoryTotal: gpu.memory_total ?? gpu.memoryTotal ?? 0,
                            temperature: gpu.temperature ?? 0
                        }));
                        gpus.sort((a: any, b: any) => a.index - b.index);
                        setGpuStats(gpus, msg.data?.timestamp ?? Date.now() / 1000);
                    } else if (msg.event === 'train/logs') {
                        addLog({
                            timestamp: new Date().toLocaleTimeString(),
                            level: msg.data.level || 'INFO',
                            message: msg.data.message
                        });
                    } else if (msg.event === 'train/started') {
                        // Assuming 10 for now if not in payload, ideally payload has config
                        startRun(msg.data.run_id, 10);
                    } else if (msg.event === 'train/completed') {
                        setStatus('completed');
                    } else if (msg.event === 'export/completed') {
                        window.dispatchEvent(new CustomEvent('vf-export-completed', { detail: msg.data }));
                    }
                } catch (e) {
                    console.error("Failed to parse WS msg", event.data);
                }
            };
        };

        connect();

        return () => {
            wsRef.current?.close();
        };
    }, []);
};
