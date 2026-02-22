import { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import { useEngineStore } from '../../store/engineStore';

const SPLASH_EXIT_MS = 280;
const SPLASH_MIN_VISIBLE_MS = 900;

export const SplashScreen = () => {
    const location = useLocation();
    const isConnected = useEngineStore((state) => state.isConnected);
    const dashboardReady = useEngineStore((state) => state.dashboardReady);
    const normalizedPath = String(location.pathname || '').trim();
    const requiresDashboardData = normalizedPath === '' || normalizedPath === '/';
    const splashReady = requiresDashboardData ? dashboardReady : (isConnected || dashboardReady);
    const [shouldRender, setShouldRender] = useState(!splashReady);
    const [isExiting, setIsExiting] = useState(false);
    const [visibleAt, setVisibleAt] = useState<number>(Date.now());

    useEffect(() => {
        if (!splashReady) {
            setShouldRender(true);
            setIsExiting(false);
            setVisibleAt(Date.now());
            return;
        }

        const elapsed = Date.now() - visibleAt;
        const remaining = Math.max(0, SPLASH_MIN_VISIBLE_MS - elapsed);
        let hideTimer = 0;
        const exitTimer = window.setTimeout(() => {
            setIsExiting(true);
            hideTimer = window.setTimeout(() => {
                setShouldRender(false);
            }, SPLASH_EXIT_MS);
        }, remaining);

        return () => {
            window.clearTimeout(exitTimer);
            if (hideTimer) {
                window.clearTimeout(hideTimer);
            }
        };
    }, [splashReady, visibleAt]);

    if (!shouldRender) return null;

    return (
        <div className={`splash-overlay ${isExiting ? 'splash-overlay-exit' : 'splash-overlay-enter'}`}>
            <div className="splash-overlay-card">
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
