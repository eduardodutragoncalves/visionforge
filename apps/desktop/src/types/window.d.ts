export {};

declare global {
    interface Window {
        ipcRenderer?: {
            on: (channel: string, listener: (...args: any[]) => void) => void;
            off: (channel: string, listener: (...args: any[]) => void) => void;
            send: (channel: string, ...args: any[]) => void;
            invoke: (channel: string, ...args: any[]) => Promise<any>;
            openDirectory: () => Promise<string | null>;
            openFiles: () => Promise<string[]>;
            openVideo: () => Promise<string | null>;
            minimize: () => Promise<void>;
            toggleMaximize: () => Promise<boolean>;
            close: () => Promise<void>;
            isMaximized: () => Promise<boolean>;
        };
    }
}
