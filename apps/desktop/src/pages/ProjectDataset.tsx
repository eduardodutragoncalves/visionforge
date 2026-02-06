import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { useEngineConnection } from '../../hooks/useEngineConnection';

declare global {
    interface Window {
        ipcRenderer: any;
    }
}

export const ProjectDataset = () => {
    const { projectId } = useParams();
    const [datasetPath, setDatasetPath] = useState<string | null>(null);
    const [scanProgress, setScanProgress] = useState<any>(null);
    const [validationReport, setValidationReport] = useState<any>(null);
    const [isValidating, setIsValidating] = useState(false);

    // Reuse engine connection to ensure WS is open
    // Ideally use a specialized hook to listen to specific events

    useEffect(() => {
        const handleMessage = (event: MessageEvent) => {
            try {
                const msg = JSON.parse(event.data);
                if (msg.event === 'dataset/scan_progress') {
                    setScanProgress(msg.data);
                } else if (msg.event === 'dataset/validate_progress') {
                    setScanProgress(msg.data); // reuse state for simplicity
                } else if (msg.event === 'dataset/validated') {
                    setValidationReport(msg.data);
                    setIsValidating(false);
                }
            } catch (e) { }
        };

        // Quick/Dirty: Add global listener or tap into hook. 
        // For this component we'll assume global WS is active and we attach a listener 
        // OR we just create a new socket if App's global one isn't exposing an event bus.
        // Let's attach to the existing socket by peeking or - better - 
        // update useEngineConnection to dispatch CustomEvents or use a store
        // For MVP, I'll direct connect a listener to the App's socket using a new connection just for this page 
        // is wasteful but easy. 
        // Or better: The `useEngineConnection` hook in App.tsx sets up the socket on mount.
        // We really need a Context or Store. 
        // I will connect a SECOND socket here for simplicity to receive events.

        const ws = new WebSocket('ws://localhost:8000/ws');
        ws.onmessage = handleMessage;

        return () => ws.close();
    }, []);

    const handleImport = async () => {
        // 1. Open Dialog
        const path = await window.ipcRenderer.openDirectory();
        if (path) {
            setDatasetPath(path);
            setIsValidating(true);

            // 2. Call Engine Import
            await fetch(`http://localhost:8000/projects/${projectId}/dataset/import`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path })
            });

            // 3. Scan & Validate
            await fetch(`http://localhost:8000/projects/${projectId}/dataset/scan`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path })
            });
        }
    };

    return (
        <div className="dataset-page animate-fade-in space-y-6">
            <header className="flex justify-between items-center mb-6">
                <div>
                    <h2 className="text-2xl font-bold">Dataset</h2>
                    <p className="text-secondary">Manage training data for {projectId}</p>
                </div>
            </header>

            <Card>
                <div className="flex justify-between items-center">
                    <div>
                        <h3 className="mb-2">Dataset Source</h3>
                        <p className="text-secondary font-mono text-sm">
                            {datasetPath || "No dataset linked yet."}
                        </p>
                    </div>
                    <Button variant="primary" onClick={handleImport} disabled={isValidating}>
                        {isValidating ? 'Validating...' : 'Import Folder'}
                    </Button>
                </div>
            </Card>

            {isValidating && (
                <Card>
                    <h3>Scanning & Validating...</h3>
                    <div className="mt-4">
                        <div className="text-sm text-secondary mb-2">
                            Processed: {scanProgress?.processed || scanProgress?.count || 0} files
                        </div>
                        {/* Fake Progress Bar */}
                        <div className="h-2 bg-gray-800 rounded overflow-hidden">
                            <div
                                className="h-full bg-accent-primary transition-all duration-300"
                                style={{ width: `${scanProgress?.progress || 0}%` }}
                            />
                        </div>
                    </div>
                </Card>
            )}

            {validationReport && (
                <div className="grid grid-cols-2 gap-6">
                    <Card glass>
                        <h3 className="mb-4 text-accent-success">Validation Passed</h3>
                        <div className="text-4xl font-bold">{validationReport.valid_count} <span className="text-base text-secondary font-normal">images</span></div>
                        <div className="mt-4 text-xs text-secondary font-mono break-all">
                            Hash: {validationReport.dataset_hash}
                        </div>
                    </Card>
                    <Card glass>
                        <h3 className="mb-4 text-accent-error">Issues Found</h3>
                        <div className="text-4xl font-bold">{validationReport.corrupted_count} <span className="text-base text-secondary font-normal">corrupted</span></div>
                        <div className="mt-4 max-h-32 overflow-y-auto">
                            {validationReport.corrupted_files.map((f: string, i: number) => (
                                <div key={i} className="text-xs text-error font-mono">{f}</div>
                            ))}
                        </div>
                    </Card>
                </div>
            )}
        </div>
    );
};
