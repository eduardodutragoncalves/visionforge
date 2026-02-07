import { useMemo } from 'react';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer
} from 'recharts';
import { useEngineStore } from '../../store/engineStore';

const LINE_COLORS = ['#2fb8ff', '#ff5c6f', '#28d17c', '#f6b73c'];

export const GpuCharts = () => {
    const { gpuTimeline, gpus } = useEngineStore((state) => ({
        gpuTimeline: state.gpuTimeline,
        gpus: state.gpus
    }));

    const chartData = useMemo(() => {
        return gpuTimeline.map((entry) => {
            const row: Record<string, number | string> = {
                time: new Date(entry.timestamp * 1000).toLocaleTimeString()
            };
            entry.gpus.forEach((gpu) => {
                row[`gpu${gpu.index}_util`] = gpu.utilization;
                row[`gpu${gpu.index}_mem`] = Math.round((gpu.memoryUsed / 1024) * 100) / 100;
            });
            return row;
        });
    }, [gpuTimeline]);

    if (chartData.length === 0 || gpus.length === 0) {
        return <div className="h-40 flex items-center justify-center text-secondary">No GPU data yet</div>;
    }

    return (
        <div className="space-y-6">
            <div>
                <div className="text-xs uppercase tracking-[0.3em] text-secondary mb-2">Utilization (%)</div>
                <div className="h-40">
                <ResponsiveContainer>
                    <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
                        <XAxis dataKey="time" stroke="#6b7280" hide />
                        <YAxis stroke="#6b7280" domain={[0, 100]} />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#0f0f13', borderColor: '#333' }}
                            itemStyle={{ color: '#fff' }}
                        />
                        {gpus.map((gpu, idx) => (
                            <Line
                                key={`util-${gpu.index}`}
                                type="monotone"
                                dataKey={`gpu${gpu.index}_util`}
                                stroke={LINE_COLORS[idx % LINE_COLORS.length]}
                                strokeWidth={2}
                                dot={false}
                                name={`GPU ${gpu.index} Util (%)`}
                            />
                        ))}
                    </LineChart>
                </ResponsiveContainer>
                </div>
            </div>

            <div>
                <div className="text-xs uppercase tracking-[0.3em] text-secondary mb-2">VRAM (GB)</div>
                <div className="h-40">
                <ResponsiveContainer>
                    <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
                        <XAxis dataKey="time" stroke="#6b7280" hide />
                        <YAxis stroke="#6b7280" />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#0f0f13', borderColor: '#333' }}
                            itemStyle={{ color: '#fff' }}
                        />
                        {gpus.map((gpu, idx) => (
                            <Line
                                key={`mem-${gpu.index}`}
                                type="monotone"
                                dataKey={`gpu${gpu.index}_mem`}
                                stroke={LINE_COLORS[idx % LINE_COLORS.length]}
                                strokeWidth={2}
                                dot={false}
                                name={`GPU ${gpu.index} VRAM (GB)`}
                            />
                        ))}
                    </LineChart>
                </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
};
