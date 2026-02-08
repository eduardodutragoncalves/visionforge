import { useEffect, useRef } from 'react';
import { useTrainingStore } from '../../store/trainingStore';
import clsx from 'clsx';

export const LogViewer = ({ runId }: { runId: string }) => {
    const logs = useTrainingStore(state => state.runs[runId]?.logs ?? []);
    const endRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        endRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [logs]);

    return (
        <div className="log-viewer glass" style={{
            height: '300px',
            overflowY: 'auto',
            padding: '16px',
            fontFamily: 'monospace',
            backgroundColor: 'rgba(0,0,0,0.3)'
        }}>
            {logs.map((log, i) => (
                <div key={i} className="log-line" style={{ fontSize: '12px', marginBottom: '4px' }}>
                    <span className="text-secondary">[{log.timestamp}]</span>
                    <span className={clsx('ml-2 font-bold', {
                        'text-blue-400': log.level === 'INFO',
                        'text-red-500': log.level === 'ERROR',
                    })}>
                        {log.level}
                    </span>:
                    <span className="ml-2 text-gray-300">{log.message}</span>
                </div>
            ))}
            <div ref={endRef} />
        </div>
    );
};
