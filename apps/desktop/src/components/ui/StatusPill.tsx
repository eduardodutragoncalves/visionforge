import clsx from 'clsx';

export type StatusType = 'idle' | 'running' | 'success' | 'error' | 'warning';

interface StatusPillProps {
    status: StatusType;
    label?: string;
    pulsing?: boolean;
    className?: string;
}

export const StatusPill = ({ status, label, pulsing, className }: StatusPillProps) => {
    return (
        <div className={clsx('status-pill', `status-${status}`, className)}>
            <div className={clsx('status-dot', pulsing && 'pulsing')} />
            <span className="status-label">{label || status}</span>
        </div>
    );
};
