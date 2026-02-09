import { contextBridge, ipcRenderer } from 'electron'

// --------- Expose some API to the Renderer process ---------
contextBridge.exposeInMainWorld('ipcRenderer', {
    on: (...args: Parameters<typeof ipcRenderer.on>) => {
        const [channel, listener] = args
        return ipcRenderer.on(channel, (event, ...args) => listener(event, ...args))
    },
    off: (...args: Parameters<typeof ipcRenderer.off>) => {
        const [channel, ...omit] = args
        return ipcRenderer.off(channel, ...omit)
    },
    send: (...args: Parameters<typeof ipcRenderer.send>) => {
        const [channel, ...omit] = args
        return ipcRenderer.send(channel, ...omit)
    },
    invoke: (...args: Parameters<typeof ipcRenderer.invoke>) => {
        const [channel, ...omit] = args
        return ipcRenderer.invoke(channel, ...omit)
    },
    openDirectory: () => ipcRenderer.invoke('dialog:openDirectory'),
    openFiles: () => ipcRenderer.invoke('dialog:openFiles'),
    openVideo: () => ipcRenderer.invoke('dialog:openVideo'),
    openPath: (targetPath: string) => ipcRenderer.invoke('shell:openPath', targetPath),
    revealPath: (targetPath: string) => ipcRenderer.invoke('shell:revealPath', targetPath),
    minimize: () => ipcRenderer.invoke('window:minimize'),
    toggleMaximize: () => ipcRenderer.invoke('window:toggle-maximize'),
    close: () => ipcRenderer.invoke('window:close'),
    isMaximized: () => ipcRenderer.invoke('window:is-maximized')
})
