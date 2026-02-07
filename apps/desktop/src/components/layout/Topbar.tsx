import { useEffect, useState } from 'react';
import { Minus, Square, Copy, X } from 'lucide-react';
import { StatusPill } from '../ui/StatusPill';

export const Topbar = () => {
    const [isMaximized, setIsMaximized] = useState(false);

    useEffect(() => {
        let mounted = true;
        const fetchState = async () => {
            if (!window.ipcRenderer?.isMaximized) return;
            const state = await window.ipcRenderer.isMaximized();
            if (mounted) {
                setIsMaximized(Boolean(state));
            }
        };

        fetchState();

        const handleMaximizeChange = (_event: unknown, value: boolean) => {
            setIsMaximized(Boolean(value));
        };

        window.ipcRenderer?.on('window:maximize-changed', handleMaximizeChange);
        return () => {
            mounted = false;
            window.ipcRenderer?.off('window:maximize-changed', handleMaximizeChange);
        };
    }, []);

    const handleMinimize = () => window.ipcRenderer?.minimize?.();
    const handleToggleMaximize = async () => {
        const nextState = await window.ipcRenderer?.toggleMaximize?.();
        if (typeof nextState === 'boolean') {
            setIsMaximized(nextState);
        }
    };
    const handleClose = () => window.ipcRenderer?.close?.();

    return (
        <header className="topbar">
            <div className="breadcrumbs">
                <span className="text-secondary">VisionForge</span>
                <span className="divider">/</span>
                <span className="current">Dashboard</span>
            </div>

            <div className="actions">
                <StatusPill status="running" label="Engine Active" pulsing />
                <div className="window-controls">
                    <button className="window-button" onClick={handleMinimize} aria-label="Minimize">
                        <Minus size={14} />
                    </button>
                    <button className="window-button" onClick={handleToggleMaximize} aria-label="Maximize">
                        {isMaximized ? <Copy size={14} /> : <Square size={14} />}
                    </button>
                    <button className="window-button window-button-close" onClick={handleClose} aria-label="Close">
                        <X size={14} />
                    </button>
                </div>
            </div>
        </header>
    );
};
