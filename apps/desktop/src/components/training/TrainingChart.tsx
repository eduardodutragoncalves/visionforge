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

export const TrainingChart = () => {
    const metrics = useTrainingStore(state => state.metrics);

    if (metrics.length === 0) return <div className="h-64 flex items-center justify-center text-secondary">No metrics yet</div>;

    return (
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
                    <Line type="monotone" dataKey="loss" stroke="#ef4444" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="accuracy" stroke="#00f0ff" strokeWidth={2} dot={false} />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
};
