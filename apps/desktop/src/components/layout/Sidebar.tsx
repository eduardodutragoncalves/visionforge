import { NavLink, useLocation, matchPath } from 'react-router-dom';
import clsx from 'clsx';
import {
    LayoutDashboard,
    FolderOpen,
    Settings,
    Activity,
    Database,
    BrainCircuit,
    Box,
    Camera
} from 'lucide-react';

export const Sidebar = () => {
    const location = useLocation();

    // Attempt to extract projectId from URL manually or via matchPath
    const match = matchPath("/projects/:projectId/*", location.pathname);
    const projectId = match?.params.projectId;

    const navLinkClass = ({ isActive }: { isActive: boolean }) =>
        clsx('nav-link', isActive && 'active');

    return (
        <aside className="sidebar glass">
            <div className="logo-container">
                <div className="logo-icon">
                    <img
                        src="/branding/ai-brain-icon.png"
                        alt="VisionForge"
                        className="logo-image"
                        onError={(event) => {
                            const el = event.currentTarget as HTMLImageElement;
                            el.style.display = 'none';
                            const fallback = el.nextElementSibling as HTMLElement | null;
                            if (fallback) fallback.style.display = 'flex';
                        }}
                    />
                    <span className="logo-fallback" style={{ display: 'none' }}>VF</span>
                </div>
            </div>

            <nav className="nav-menu">
                <NavLink to="/" className={navLinkClass} title="Dashboard">
                    <LayoutDashboard size={24} />
                    {/* Only show label on hover or if expanded? Keeping minimal icon-only for now based on previous code */}
                </NavLink>

                <div className="my-4 border-b border-white/10 mx-4" />

                <NavLink to="/projects" className={navLinkClass} title="All Projects">
                    <FolderOpen size={24} />
                </NavLink>

                {/* Project Specific Links */}
                {projectId && (
                    <div className="flex flex-col gap-2 mt-4 p-2 bg-white/5 rounded-lg mx-2 border border-white/10 animate-fade-in">
                        <NavLink to={`/projects/${projectId}/dataset`} className={navLinkClass} title="Dataset">
                            <Database size={20} />
                        </NavLink>
                        <NavLink to={`/projects/${projectId}/train/setup`} className={navLinkClass} title="Train">
                            <BrainCircuit size={20} />
                        </NavLink>
                        <NavLink to={`/projects/${projectId}/test`} className={navLinkClass} title="Test">
                            <Camera size={20} />
                        </NavLink>
                        <NavLink to={`/projects/${projectId}/models`} className={navLinkClass} title="Models">
                            <Box size={20} />
                        </NavLink>
                    </div>
                )}

                <div className="flex-1" />

                <NavLink to="/diagnostics" className={navLinkClass} title="Diagnostics">
                    <Activity size={24} />
                </NavLink>
                <NavLink to="/settings" className={navLinkClass} title="Settings">
                    <Settings size={24} />
                </NavLink>
            </nav>

            <div className="sidebar-footer">
                <div className="version">v0.1</div>
            </div>
        </aside>
    );
};
