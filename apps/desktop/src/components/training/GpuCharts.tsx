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

type GpuChartsProps = {
    visibleGpuIndices?: number[];
    compact?: boolean;
    chartHeight?: number;
};

export const GpuCharts = ({ visibleGpuIndices, compact = false, chartHeight }: GpuChartsProps) => {
    const { gpuTimeline, gpus } = useEngineStore((state) => ({
        gpuTimeline: state.gpuTimeline,
        gpus: state.gpus
    }));

    const visibleGpuSet = useMemo(() => {
        if (!visibleGpuIndices || visibleGpuIndices.length === 0) return null;
        return new Set(visibleGpuIndices.map((value) => Number(value)).filter((value) => Number.isFinite(value)));
    }, [visibleGpuIndices]);

    const visibleGpus = useMemo(() => {
        if (!visibleGpuSet || visibleGpuSet.size === 0) return gpus;
        return gpus.filter((gpu) => visibleGpuSet.has(gpu.index));
    }, [gpus, visibleGpuSet]);

    const chartData = useMemo(() => {
        return gpuTimeline.map((entry) => {
            const row: Record<string, number | string> = {
                time: new Date(entry.timestamp * 1000).toLocaleTimeString()
            };
            entry.gpus
                .filter((gpu) => !visibleGpuSet || visibleGpuSet.has(gpu.index))
                .forEach((gpu) => {
                const used = Math.round((gpu.memoryUsed / 1024) * 100) / 100;
                const total = Math.round((gpu.memoryTotal / 1024) * 100) / 100;
                row[`gpu${gpu.index}_util`] = gpu.utilization;
                row[`gpu${gpu.index}_mem`] = used;
                row[`gpu${gpu.index}_mem_total`] = total;
                row[`gpu${gpu.index}_mem_pct`] = total > 0 ? Math.round((used / total) * 1000) / 10 : 0;
                row[`gpu${gpu.index}_temp`] = gpu.temperature;
            });
            return row;
        });
    }, [gpuTimeline, visibleGpuSet]);

    const maxMemTotal = useMemo(() => {
        const totals = visibleGpus.map((gpu) => gpu.memoryTotal);
        if (totals.length === 0) return 0;
        return Math.max(...totals) / 1024;
    }, [visibleGpus]);

    const latestSnapshot = useMemo(() => {
        if (visibleGpus.length === 0) return [];
        return visibleGpus.map((gpu) => {
            const used = Math.round((gpu.memoryUsed / 1024) * 100) / 100;
            const total = Math.round((gpu.memoryTotal / 1024) * 100) / 100;
            const pct = total > 0 ? Math.round((used / total) * 1000) / 10 : 0;
            return { index: gpu.index, used, total, pct, temp: gpu.temperature };
        });
    }, [visibleGpus]);

    const maxTemp = useMemo(() => {
        if (visibleGpus.length === 0) return 100;
        const temps = visibleGpus.map((gpu) => gpu.temperature || 0);
        const max = Math.max(...temps, 100);
        return Math.ceil(max / 10) * 10;
    }, [visibleGpus]);

    if (chartData.length === 0 || visibleGpus.length === 0) {
        return <div className="h-40 flex items-center justify-center text-secondary">No GPU data yet</div>;
    }

    const resolvedChartHeight = Math.max(70, chartHeight ?? (compact ? 90 : 140));

    return (
        <div className={compact ? 'space-y-4' : 'space-y-6'}>
            <div className={`flex flex-wrap gap-3 uppercase tracking-[0.3em] text-secondary ${compact ? 'text-[10px]' : 'text-xs'}`}>
                {latestSnapshot.map((gpu) => (
                    <div
                        key={`gpu-summary-${gpu.index}`}
                        className="border border-graphite-700/70 bg-graphite-900 px-3 py-2 rounded"
                    >
                        GPU {gpu.index}: <span className="text-bone">{gpu.used} / {gpu.total} GB</span> ({gpu.pct}%)
                        <span className="ml-2 text-secondary">· {gpu.temp}°C</span>
                    </div>
                ))}
            </div>
            <div>
                <div className="text-xs uppercase tracking-[0.3em] text-secondary mb-2">Utilization (%)</div>
                <div className={`flex flex-wrap gap-3 uppercase tracking-[0.3em] text-secondary mb-2 ${compact ? 'text-[9px]' : 'text-[10px]'}`}>
                    {visibleGpus.map((gpu, idx) => (
                        <div key={`util-legend-${gpu.index}`} className="flex items-center gap-2">
                            <span className="h-2 w-2 rounded-full" style={{ backgroundColor: LINE_COLORS[idx % LINE_COLORS.length] }} />
                            <span>GPU {gpu.index}</span>
                        </div>
                    ))}
                </div>
                <div style={{ height: resolvedChartHeight, minHeight: resolvedChartHeight }}>
                <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={resolvedChartHeight}>
                    <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
                        <XAxis dataKey="time" stroke="#6b7280" hide />
                        <YAxis stroke="#6b7280" domain={[0, 100]} />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#0f0f13', borderColor: '#333' }}
                            itemStyle={{ color: '#fff' }}
                        />
                        {visibleGpus.map((gpu, idx) => (
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
                <div className={`flex flex-wrap gap-3 uppercase tracking-[0.3em] text-secondary mb-2 ${compact ? 'text-[9px]' : 'text-[10px]'}`}>
                    {visibleGpus.map((gpu, idx) => (
                        <div key={`mem-legend-${gpu.index}`} className="flex items-center gap-2">
                            <span className="h-2 w-2 rounded-full" style={{ backgroundColor: LINE_COLORS[idx % LINE_COLORS.length] }} />
                            <span>GPU {gpu.index}</span>
                        </div>
                    ))}
                </div>
                <div style={{ height: resolvedChartHeight, minHeight: resolvedChartHeight }}>
                <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={resolvedChartHeight}>
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
                        {visibleGpus.map((gpu, idx) => (
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

            <div>
                <div className="text-xs uppercase tracking-[0.3em] text-secondary mb-2">Temperature (°C)</div>
                <div className={`flex flex-wrap gap-3 uppercase tracking-[0.3em] text-secondary mb-2 ${compact ? 'text-[9px]' : 'text-[10px]'}`}>
                    {visibleGpus.map((gpu, idx) => (
                        <div key={`temp-legend-${gpu.index}`} className="flex items-center gap-2">
                            <span className="h-2 w-2 rounded-full" style={{ backgroundColor: LINE_COLORS[idx % LINE_COLORS.length] }} />
                            <span>GPU {gpu.index}</span>
                        </div>
                    ))}
                </div>
                <div style={{ height: resolvedChartHeight, minHeight: resolvedChartHeight }}>
                <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={resolvedChartHeight}>
                    <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
                        <XAxis dataKey="time" stroke="#6b7280" hide />
                        <YAxis stroke="#6b7280" domain={[0, maxTemp]} />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#0f0f13', borderColor: '#333' }}
                            itemStyle={{ color: '#fff' }}
                        />
                        {visibleGpus.map((gpu, idx) => (
                            <Line
                                key={`temp-${gpu.index}`}
                                type="monotone"
                                dataKey={`gpu${gpu.index}_temp`}
                                stroke={LINE_COLORS[idx % LINE_COLORS.length]}
                                strokeWidth={2}
                                dot={false}
                                name={`GPU ${gpu.index} Temp (°C)`}
                            />
                        ))}
                    </LineChart>
                </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
};
