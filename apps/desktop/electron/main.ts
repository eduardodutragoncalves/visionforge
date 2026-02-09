import { app, BrowserWindow, shell, ipcMain, dialog } from 'electron'
import path from 'node:path'
import { release } from 'node:os'
import http from 'node:http'
import fs from 'node:fs'
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'

if (release().startsWith('6.1')) {
    app.disableHardwareAcceleration()
}
if (process.platform === 'win32') {
    app.setAppUserModelId(app.getName())
}

const DIST_ELECTRON = path.join(__dirname, '../dist-electron')
const DIST = path.join(__dirname, '../dist')
const PUBLIC = app.isPackaged ? DIST : path.join(__dirname, '../public')

process.env.DIST_ELECTRON = DIST_ELECTRON
process.env.DIST = DIST
process.env.PUBLIC = PUBLIC

const VITE_DEV_SERVER_URL = process.env['VITE_DEV_SERVER_URL']
const DEFAULT_DEV_SERVER_URL = 'http://127.0.0.1:5173/'
const isDev = !app.isPackaged

let win: BrowserWindow | null = null
let engineProcess: ChildProcessWithoutNullStreams | null = null
let engineOwned = false

const ENGINE_HEALTH_URL = process.env.VISIONFORGE_ENGINE_URL || 'http://127.0.0.1:8000/health'
const ENGINE_PYTHON = process.env.VISIONFORGE_PYTHON || 'python'
const resolveEngineCwd = () => {
    if (process.env.VISIONFORGE_ENGINE_CWD) {
        return process.env.VISIONFORGE_ENGINE_CWD
    }
    const repoRootCandidate = path.resolve(__dirname, '../../..')
    const engineEntry = path.join(repoRootCandidate, 'engine', 'main.py')
    if (fs.existsSync(engineEntry)) {
        return repoRootCandidate
    }
    return process.cwd()
}

const ENGINE_CWD = resolveEngineCwd()

const isEngineRunning = () => new Promise<boolean>((resolve) => {
    const req = http.get(ENGINE_HEALTH_URL, (res) => {
        resolve(res.statusCode === 200)
        res.resume()
    })
    req.on('error', () => resolve(false))
    req.setTimeout(500, () => {
        req.destroy()
        resolve(false)
    })
})

const startEngine = async () => {
    const running = await isEngineRunning()
    if (running) {
        engineOwned = false
        return
    }

    const args = ['-m', 'uvicorn', 'engine.main:app', '--port', '8000']
    if (process.env.VITE_DEV_SERVER_URL) {
        args.push('--reload')
    }

    engineProcess = spawn(ENGINE_PYTHON, args, {
        cwd: ENGINE_CWD,
        env: process.env,
        stdio: 'pipe',
        detached: process.platform !== 'win32'
    })
    engineOwned = true

    engineProcess.stdout.on('data', (data) => {
        console.log(`[engine] ${data.toString().trim()}`)
    })
    engineProcess.stderr.on('data', (data) => {
        console.error(`[engine] ${data.toString().trim()}`)
    })
    engineProcess.on('exit', (code) => {
        console.log(`[engine] exited with code ${code ?? 'unknown'}`)
        engineProcess = null
    })
}

const stopEngine = () => {
    if (!engineProcess || !engineOwned) return

    const pid = engineProcess.pid
    if (!pid) return

    if (process.platform === 'win32') {
        spawn('taskkill', ['/pid', String(pid), '/t', '/f'])
    } else {
        try {
            process.kill(-pid, 'SIGTERM')
        } catch (err) {
            try {
                process.kill(pid, 'SIGTERM')
            } catch {
                // ignore
            }
        }
    }
}

async function handleOpenDirectory() {
    const { canceled, filePaths } = await dialog.showOpenDialog({
        properties: ['openDirectory', 'createDirectory']
    })
    if (canceled) {
        return null
    }
    return filePaths[0]
}

async function handleOpenFiles() {
    const { canceled, filePaths } = await dialog.showOpenDialog({
        properties: ['openFile', 'multiSelections'],
        filters: [{ name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'bmp', 'webp'] }]
    })
    if (canceled) {
        return []
    }
    return filePaths
}

async function handleOpenVideo() {
    const { canceled, filePaths } = await dialog.showOpenDialog({
        properties: ['openFile'],
        filters: [{ name: 'Videos', extensions: ['mp4', 'mov', 'avi', 'mkv'] }]
    })
    if (canceled) {
        return null
    }
    return filePaths[0]
}

async function handleOpenPath(targetPath: string) {
    if (!targetPath || typeof targetPath !== 'string') {
        return { ok: false, message: 'Invalid path' }
    }
    const result = await shell.openPath(targetPath)
    if (result) {
        return { ok: false, message: result }
    }
    return { ok: true }
}

async function handleRevealPath(targetPath: string) {
    if (!targetPath || typeof targetPath !== 'string') {
        return { ok: false, message: 'Invalid path' }
    }
    shell.showItemInFolder(targetPath)
    return { ok: true }
}

function createWindow() {
    win = new BrowserWindow({
        icon: path.join(PUBLIC, 'branding', 'ai-brain-icon-256.png'),
        width: 1200,
        height: 800,
        frame: false,
        show: false,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
        },
        vibrancy: 'fullscreen-ui',
        backgroundMaterial: 'acrylic',
        backgroundColor: '#00000000',
    })

    win.on('closed', () => {
        win = null
    })

    win.webContents.setWindowOpenHandler(({ url }) => {
        shell.openExternal(url)
        return { action: 'deny' }
    })

    win.webContents.on('did-finish-load', () => {
        win?.webContents.send('main-process-message', new Date().toLocaleString())
    })
    win.webContents.on('did-fail-load', (_event, code, desc, url) => {
        console.error(`[renderer] failed to load (${code}): ${desc} -> ${url}`)
    })
    win.webContents.on('render-process-gone', (_event, details) => {
        console.error(`[renderer] process gone: ${details.reason}`)
    })
    win.webContents.on('console-message', (_event, level, message, line, sourceId) => {
        const levelName = ['log', 'warn', 'error'][level] ?? 'log'
        console.log(`[renderer:${levelName}] ${message} (${sourceId}:${line})`)
    })
    win.once('ready-to-show', () => {
        win?.maximize()
        win?.show()
        if (process.env.VITE_OPEN_DEVTOOLS === '1') {
            win?.webContents.openDevTools({ mode: 'detach' })
        }
    })

    const notifyMaximize = () => {
        win?.webContents.send('window:maximize-changed', win.isMaximized())
    }

    win.on('maximize', notifyMaximize)
    win.on('unmaximize', notifyMaximize)

    const devServerUrl = VITE_DEV_SERVER_URL || (isDev ? DEFAULT_DEV_SERVER_URL : '')
    if (devServerUrl) {
        win.loadURL(devServerUrl).catch((err) => {
            console.error('[renderer] failed to load dev server', err)
        })
    } else {
        win.loadFile(path.join(DIST, 'index.html')).catch((err) => {
            console.error('[renderer] failed to load index.html', err)
        })
    }
}

app.whenReady().then(() => {
    ipcMain.handle('dialog:openDirectory', handleOpenDirectory)
    ipcMain.handle('dialog:openFiles', handleOpenFiles)
    ipcMain.handle('dialog:openVideo', handleOpenVideo)
    ipcMain.handle('shell:openPath', (_event, targetPath: string) => handleOpenPath(targetPath))
    ipcMain.handle('shell:revealPath', (_event, targetPath: string) => handleRevealPath(targetPath))
    ipcMain.handle('window:minimize', () => win?.minimize())
    ipcMain.handle('window:toggle-maximize', () => {
        if (!win) return false
        if (win.isMaximized()) {
            win.unmaximize()
            return false
        }
        win.maximize()
        return true
    })
    ipcMain.handle('window:close', () => win?.close())
    ipcMain.handle('window:is-maximized', () => win?.isMaximized() ?? false)
    createWindow()
    startEngine()
    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow()
        }
    })
})

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        stopEngine()
        app.quit()
    }
})

app.on('before-quit', () => {
    stopEngine()
})
