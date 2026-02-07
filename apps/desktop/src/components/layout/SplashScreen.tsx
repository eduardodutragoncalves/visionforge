import { useEngineStore } from '../../store/engineStore';

export const SplashScreen = () => {
    const isConnected = useEngineStore((state) => state.isConnected);

    if (isConnected) return null;

    return (
        <div className="splash-inline">
            <div className="splash-inline-card">
                <div className="splash-inline-title">Connecting to engine</div>
                <div className="splash-loader">
                    <span className="loader" />
                    <span className="splash-status">Waiting for backend...</span>
                </div>
                <div className="splash-progress" aria-hidden="true" />
            </div>
        </div>
    );
};
