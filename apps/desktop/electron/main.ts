import { app, BrowserWindow, shell, ipcMain, dialog } from 'electron'
import path from 'node:path'
import { release } from 'node:os'

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

let win: BrowserWindow | null = null

async function handleOpenDirectory() {
    const { canceled, filePaths } = await dialog.showOpenDialog({
        properties: ['openDirectory', 'createDirectory']
    })
    if (canceled) {
        return null
    }
    return filePaths[0]
}

function createWindow() {
    win = new BrowserWindow({
        icon: path.join(PUBLIC, 'electron-vite.svg'),
        width: 1200,
        height: 800,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
        },
        titleBarStyle: 'hidden',
        titleBarOverlay: {
            color: 'rgba(0,0,0,0)',
            symbolColor: '#ffffff'
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

    if (VITE_DEV_SERVER_URL) {
        win.loadURL(VITE_DEV_SERVER_URL)
    } else {
        win.loadFile(path.join(DIST, 'index.html'))
    }
}

app.whenReady().then(() => {
    ipcMain.handle('dialog:openDirectory', handleOpenDirectory)
    createWindow()
    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow()
        }
    })
})

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit()
    }
})
