import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Topbar } from './Topbar';
import { SplashScreen } from './SplashScreen';

export const AppShell = () => {
    return (
        <div className="app-shell">
            <Sidebar />
            <div className="content-wrapper">
                <Topbar />
                <SplashScreen />
                <main className="page-content">
                    <Outlet />
                </main>
            </div>
        </div>
    );
};
