import { StatusPill } from '../ui/StatusPill';

export const Topbar = () => {
    return (
        <header className="topbar">
            <div className="breadcrumbs">
                <span className="text-secondary">VisionForge</span>
                <span className="divider">/</span>
                <span className="current">Dashboard</span>
            </div>

            <div className="actions">
                <StatusPill status="running" label="Engine Active" pulsing />
            </div>
        </header>
    );
};
