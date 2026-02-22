import { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import { useEngineStore } from '../../store/engineStore';

export const SplashScreen = () => {
    const location = useLocation();
    const isConnected = useEngineStore((state) => state.isConnected);
    const dashboardReady = useEngineStore((state) => state.dashboardReady);
    const normalizedPath = String(location.pathname || '').trim();
    const requiresDashboardData = normalizedPath === '' || normalizedPath === '/';
    const splashReady = requiresDashboardData ? dashboardReady : (isConnected || dashboardReady);
    const [shouldRender, setShouldRender] = useState(!splashReady);
    const [isExiting, setIsExiting] = useState(false);

    useEffect(() => {
        if (!splashReady) {
            setShouldRender(true);
            setIsExiting(false);
            return;
        }

        setIsExiting(true);
        const timer = window.setTimeout(() => {
            setShouldRender(false);
        }, 280);

        return () => window.clearTimeout(timer);
    }, [splashReady]);

    if (!shouldRender) return null;

    return (
        <div className={`splash-inline ${isExiting ? 'splash-inline-exit' : 'splash-inline-enter'}`}>
            <div className="splash-inline-card">
                <div className="splash-brand">
                    <img src="/branding/ai-brain-icon.png" alt="VisionForge" className="splash-brand-image" />
                </div>
                <div className="splash-inline-title">Connecting to engine</div>
                <div className="splash-loader">
                    <span className="loader" />
                    <span className="splash-status">
                        {requiresDashboardData
                            ? 'Loading dashboard data...'
                            : (isConnected ? 'Engine connected.' : 'Waiting for backend...')}
                    </span>
                </div>
                <div className="splash-progress" aria-hidden="true" />
            </div>
        </div>
    );
};
