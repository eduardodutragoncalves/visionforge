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

interface TrainingChartProps {
    runId: string;
    compact?: boolean;
    secondaryKey?: 'accuracy' | 'map50';
    secondaryLabel?: string;
}

export const TrainingChart = ({
    runId,
    compact = false,
    secondaryKey = 'accuracy',
    secondaryLabel = 'Accuracy'
}: TrainingChartProps) => {
    const metrics = useTrainingStore(state => state.runs[runId]?.metrics ?? []);
    const series = [
        { key: 'loss', label: 'Loss', color: '#ef4444' },
        { key: secondaryKey, label: secondaryLabel, color: '#00f0ff' }
    ];

    if (metrics.length === 0) {
        return (
            <div className={`flex items-center justify-center text-secondary ${compact ? 'h-40' : 'h-64'}`}>
                No metrics yet
            </div>
        );
    }

    return (
        <div className="flex h-full min-h-0 flex-col gap-2">
            <div className={`flex flex-wrap gap-3 uppercase tracking-[0.3em] text-secondary ${compact ? 'text-[10px]' : 'text-xs'}`}>
                {series.map((item) => (
                    <div key={item.key} className="flex items-center gap-2">
                        <span className="h-2 w-2 rounded-full" style={{ backgroundColor: item.color }} />
                        <span>{item.label}</span>
                    </div>
                ))}
            </div>
            <div
                className="flex-1 min-h-0"
                style={{ width: '100%', minHeight: compact ? 160 : 240 }}
            >
                <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={compact ? 160 : 240}>
                    <LineChart data={metrics}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
                        <XAxis dataKey="epoch" stroke="#6b7280" />
                        <YAxis stroke="#6b7280" />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#0f0f13', borderColor: '#333' }}
                            itemStyle={{ color: '#fff' }}
                        />
                        {series.map((item) => (
                            <Line
                                key={item.key}
                                type="monotone"
                                dataKey={item.key}
                                stroke={item.color}
                                strokeWidth={2}
                                dot={false}
                                connectNulls
                            />
                        ))}
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};
