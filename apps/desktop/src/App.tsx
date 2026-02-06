import { Routes, Route, Navigate } from 'react-router-dom'
import { AppShell } from './components/layout/AppShell'
import { Dashboard } from './pages/Dashboard'
import { Projects } from './pages/Projects'
import { Settings } from './pages/Settings'
import { Diagnostics } from './pages/Diagnostics'
import { LiveRun } from './pages/LiveRun'
import { ProjectDataset } from './pages/ProjectDataset'
import { TrainSetup } from './pages/TrainSetup'
import { ProjectModels } from './pages/ProjectModels'
import { ProjectTest } from './pages/ProjectTest'
import { useEngineConnection } from './hooks/useEngineConnection'

function App() {
    useEngineConnection();

    return (
        <Routes>
            <Route element={<AppShell />}>
                <Route path="/" element={<Dashboard />} />
                <Route path="/projects" element={<Projects />} />
                <Route path="/projects/:projectId/dataset" element={<ProjectDataset />} />
                <Route path="/projects/:projectId/train/setup" element={<TrainSetup />} />
                <Route path="/projects/:projectId/train/live/:runId" element={<LiveRun />} />
                <Route path="/projects/:projectId/test" element={<ProjectTest />} />
                <Route path="/projects/:projectId/models" element={<ProjectModels />} />
                <Route path="/settings" element={<Settings />} />
                <Route path="/diagnostics" element={<Diagnostics />} />
                <Route path="*" element={<Navigate to="/" replace />} />
            </Route>
        </Routes>
    )
}

export default App
