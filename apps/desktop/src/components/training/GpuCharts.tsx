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
                const used = Math.round((gpu.memoryUsed / 1024) * 100) / 100;
                const total = Math.round((gpu.memoryTotal / 1024) * 100) / 100;
                row[`gpu${gpu.index}_util`] = gpu.utilization;
                row[`gpu${gpu.index}_mem`] = used;
                row[`gpu${gpu.index}_mem_total`] = total;
                row[`gpu${gpu.index}_mem_pct`] = total > 0 ? Math.round((used / total) * 1000) / 10 : 0;
            });
            return row;
        });
    }, [gpuTimeline]);

    const maxMemTotal = useMemo(() => {
        const totals = gpus.map((gpu) => gpu.memoryTotal);
        if (totals.length === 0) return 0;
        return Math.max(...totals) / 1024;
    }, [gpus]);

    const latestSnapshot = useMemo(() => {
        if (gpus.length === 0) return [];
        return gpus.map((gpu) => {
            const used = Math.round((gpu.memoryUsed / 1024) * 100) / 100;
            const total = Math.round((gpu.memoryTotal / 1024) * 100) / 100;
            const pct = total > 0 ? Math.round((used / total) * 1000) / 10 : 0;
            return { index: gpu.index, used, total, pct };
        });
    }, [gpus]);

    if (chartData.length === 0 || gpus.length === 0) {
        return <div className="h-40 flex items-center justify-center text-secondary">No GPU data yet</div>;
    }

    return (
        <div className="space-y-6">
            <div className="flex flex-wrap gap-3 text-xs uppercase tracking-[0.3em] text-secondary">
                {latestSnapshot.map((gpu) => (
                    <div
                        key={`gpu-summary-${gpu.index}`}
                        className="border border-graphite-700/70 bg-graphite-900 px-3 py-2 rounded"
                    >
                        GPU {gpu.index}: <span className="text-bone">{gpu.used} / {gpu.total} GB</span> ({gpu.pct}%)
                    </div>
                ))}
            </div>
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
                        <YAxis stroke="#6b7280" domain={[0, Math.max(1, Math.ceil(maxMemTotal))]} />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#0f0f13', borderColor: '#333' }}
                            itemStyle={{ color: '#fff' }}
                            formatter={(value, name, props) => {
                                if (typeof name !== 'string' || !name.includes('VRAM')) {
                                    return value;
                                }
                                const payload = props?.payload ?? {};
                                const match = String(props?.dataKey || '').match(/gpu(\\d+)_mem/);
                                if (!match) return value;
                                const idx = match[1];
                                const total = payload[`gpu${idx}_mem_total`];
                                const pct = payload[`gpu${idx}_mem_pct`];
                                if (typeof total === 'number' && typeof pct === 'number') {
                                    return [`${value} GB / ${total} GB (${pct}%)`, name];
                                }
                                return value;
                            }}
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
