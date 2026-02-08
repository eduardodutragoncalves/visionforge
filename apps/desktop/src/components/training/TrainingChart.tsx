import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer
} from 'recharts';
import { useTrainingStore } from '../../store/trainingStore';

const SERIES = [
    { key: 'loss', label: 'Loss', color: '#ef4444' },
    { key: 'accuracy', label: 'Accuracy', color: '#00f0ff' }
];

export const TrainingChart = ({ runId }: { runId: string }) => {
    const metrics = useTrainingStore(state => state.runs[runId]?.metrics ?? []);

    if (metrics.length === 0) return <div className="h-64 flex items-center justify-center text-secondary">No metrics yet</div>;

    return (
        <div className="space-y-2">
            <div className="flex flex-wrap gap-3 text-xs uppercase tracking-[0.3em] text-secondary">
                {SERIES.map((item) => (
                    <div key={item.key} className="flex items-center gap-2">
                        <span className="h-2 w-2 rounded-full" style={{ backgroundColor: item.color }} />
                        <span>{item.label}</span>
                    </div>
                ))}
            </div>
            <div style={{ width: '100%', height: 300 }}>
                <ResponsiveContainer>
                    <LineChart data={metrics}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
                        <XAxis dataKey="epoch" stroke="#6b7280" />
                        <YAxis stroke="#6b7280" />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#0f0f13', borderColor: '#333' }}
                            itemStyle={{ color: '#fff' }}
                        />
                        {SERIES.map((item) => (
                            <Line
                                key={item.key}
                                type="monotone"
                                dataKey={item.key}
                                stroke={item.color}
                                strokeWidth={2}
                                dot={false}
                            />
                        ))}
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};
