"use strict";
const electron = require("electron");
const path = require("node:path");
const node_os = require("node:os");
const http = require("node:http");
const fs = require("node:fs");
const node_child_process = require("node:child_process");
if (node_os.release().startsWith("6.1")) {
  electron.app.disableHardwareAcceleration();
}
if (process.platform === "win32") {
  electron.app.setAppUserModelId(electron.app.getName());
}
const DIST_ELECTRON = path.join(__dirname, "../dist-electron");
const DIST = path.join(__dirname, "../dist");
const PUBLIC = electron.app.isPackaged ? DIST : path.join(__dirname, "../public");
process.env.DIST_ELECTRON = DIST_ELECTRON;
process.env.DIST = DIST;
process.env.PUBLIC = PUBLIC;
const VITE_DEV_SERVER_URL = process.env["VITE_DEV_SERVER_URL"];
const DEFAULT_DEV_SERVER_URL = "http://127.0.0.1:5173/";
const isDev = !electron.app.isPackaged;
let win = null;
let engineProcess = null;
let engineOwned = false;
const ENGINE_HEALTH_URL = process.env.VISIONFORGE_ENGINE_URL || "http://127.0.0.1:8000/health";
const ENGINE_PYTHON = process.env.VISIONFORGE_PYTHON || "python";
const ENGINE_PORT = process.env.VISIONFORGE_ENGINE_PORT || "8000";
const resolveEngineCwd = () => {
  if (process.env.VISIONFORGE_ENGINE_CWD) {
    return process.env.VISIONFORGE_ENGINE_CWD;
  }
  const repoRootCandidate = path.resolve(__dirname, "../../..");
  const engineEntry = path.join(repoRootCandidate, "engine", "main.py");
  if (fs.existsSync(engineEntry)) {
    return repoRootCandidate;
  }
  return process.cwd();
};
const ENGINE_CWD = resolveEngineCwd();
const isEngineRunning = () => new Promise((resolve) => {
  const req = http.get(ENGINE_HEALTH_URL, (res) => {
    resolve(res.statusCode === 200);
    res.resume();
  });
  req.on("error", () => resolve(false));
  req.setTimeout(2500, () => {
    req.destroy();
    resolve(false);
  });
});
const cleanupUnhealthyEngine = async () => {
  if (process.platform === "win32") return;
  await new Promise((resolve) => {
    const killer = node_child_process.spawn("pkill", ["-f", `uvicorn engine.main:app --port ${ENGINE_PORT}`], {
      stdio: "ignore"
    });
    killer.on("error", () => resolve());
    killer.on("exit", () => resolve());
  });
};
const startEngine = async () => {
  const running = await isEngineRunning();
  if (running) {
    engineOwned = false;
    return;
  }
  await cleanupUnhealthyEngine();
  const becameHealthyAfterCleanup = await isEngineRunning();
  if (becameHealthyAfterCleanup) {
    engineOwned = false;
    return;
  }
  const args = ["-m", "uvicorn", "engine.main:app", "--port", ENGINE_PORT];
  engineProcess = node_child_process.spawn(ENGINE_PYTHON, args, {
    cwd: ENGINE_CWD,
    env: process.env,
    stdio: "pipe",
    detached: process.platform !== "win32"
  });
  engineOwned = true;
  engineProcess.stdout.on("data", (data) => {
    console.log(`[engine] ${data.toString().trim()}`);
  });
  engineProcess.stderr.on("data", (data) => {
    console.error(`[engine] ${data.toString().trim()}`);
  });
  engineProcess.on("exit", (code) => {
    console.log(`[engine] exited with code ${code ?? "unknown"}`);
    engineProcess = null;
  });
};
const stopEngine = () => {
  if (!engineProcess || !engineOwned) return;
  const pid = engineProcess.pid;
  if (!pid) return;
  if (process.platform === "win32") {
    node_child_process.spawn("taskkill", ["/pid", String(pid), "/t", "/f"]);
  } else {
    try {
      process.kill(-pid, "SIGTERM");
    } catch (err) {
      try {
        process.kill(pid, "SIGTERM");
      } catch {
      }
    }
  }
};
async function handleOpenDirectory() {
  const { canceled, filePaths } = await electron.dialog.showOpenDialog({
    properties: ["openDirectory", "createDirectory"]
  });
  if (canceled) {
    return null;
  }
  return filePaths[0];
}
async function handleOpenFiles() {
  const { canceled, filePaths } = await electron.dialog.showOpenDialog({
    properties: ["openFile", "multiSelections"],
    filters: [{ name: "Images", extensions: ["jpg", "jpeg", "png", "bmp", "webp"] }]
  });
  if (canceled) {
    return [];
  }
  return filePaths;
}
async function handleOpenVideo() {
  const { canceled, filePaths } = await electron.dialog.showOpenDialog({
    properties: ["openFile"],
    filters: [{ name: "Videos", extensions: ["mp4", "mov", "avi", "mkv"] }]
  });
  if (canceled) {
    return null;
  }
  return filePaths[0];
}
async function handleOpenPath(targetPath) {
  if (!targetPath || typeof targetPath !== "string") {
    return { ok: false, message: "Invalid path" };
  }
  const result = await electron.shell.openPath(targetPath);
  if (result) {
    return { ok: false, message: result };
  }
  return { ok: true };
}
async function handleRevealPath(targetPath) {
  if (!targetPath || typeof targetPath !== "string") {
    return { ok: false, message: "Invalid path" };
  }
  electron.shell.showItemInFolder(targetPath);
  return { ok: true };
}
async function handleReadTextFile(targetPath) {
  if (!targetPath || typeof targetPath !== "string") {
    return { ok: false, message: "Invalid path" };
  }
  try {
    const resolved = path.resolve(targetPath);
    const stats = await fs.promises.stat(resolved);
    if (!stats.isFile()) {
      return { ok: false, message: "Path is not a file" };
    }
    if (stats.size > 2 * 1024 * 1024) {
      return { ok: false, message: "File too large to preview in app" };
    }
    const content = await fs.promises.readFile(resolved, "utf-8");
    return { ok: true, content };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { ok: false, message };
  }
}
function createWindow() {
  win = new electron.BrowserWindow({
    icon: path.join(PUBLIC, "branding", "ai-brain-icon-256.png"),
    width: 1200,
    height: 800,
    frame: false,
    show: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.js")
    },
    vibrancy: "fullscreen-ui",
    backgroundMaterial: "acrylic",
    backgroundColor: "#00000000"
  });
  win.on("closed", () => {
    win = null;
  });
  win.webContents.setWindowOpenHandler(({ url }) => {
    electron.shell.openExternal(url);
    return { action: "deny" };
  });
  win.webContents.on("did-finish-load", () => {
    win?.webContents.send("main-process-message", (/* @__PURE__ */ new Date()).toLocaleString());
  });
  win.webContents.on("did-fail-load", (_event, code, desc, url) => {
    console.error(`[renderer] failed to load (${code}): ${desc} -> ${url}`);
  });
  win.webContents.on("render-process-gone", (_event, details) => {
    console.error(`[renderer] process gone: ${details.reason}`);
  });
  win.webContents.on("console-message", (_event, level, message, line, sourceId) => {
    const levelName = ["log", "warn", "error"][level] ?? "log";
    console.log(`[renderer:${levelName}] ${message} (${sourceId}:${line})`);
  });
  win.once("ready-to-show", () => {
    win?.maximize();
    win?.show();
    if (process.env.VITE_OPEN_DEVTOOLS === "1") {
      win?.webContents.openDevTools({ mode: "detach" });
    }
  });
  const notifyMaximize = () => {
    win?.webContents.send("window:maximize-changed", win.isMaximized());
  };
  win.on("maximize", notifyMaximize);
  win.on("unmaximize", notifyMaximize);
  const devServerUrl = VITE_DEV_SERVER_URL || (isDev ? DEFAULT_DEV_SERVER_URL : "");
  if (devServerUrl) {
    win.loadURL(devServerUrl).catch((err) => {
      console.error("[renderer] failed to load dev server", err);
    });
  } else {
    win.loadFile(path.join(DIST, "index.html")).catch((err) => {
      console.error("[renderer] failed to load index.html", err);
    });
  }
}
electron.app.whenReady().then(() => {
  electron.ipcMain.handle("dialog:openDirectory", handleOpenDirectory);
  electron.ipcMain.handle("dialog:openFiles", handleOpenFiles);
  electron.ipcMain.handle("dialog:openVideo", handleOpenVideo);
  electron.ipcMain.handle("shell:openPath", (_event, targetPath) => handleOpenPath(targetPath));
  electron.ipcMain.handle("shell:revealPath", (_event, targetPath) => handleRevealPath(targetPath));
  electron.ipcMain.handle("fs:readTextFile", (_event, targetPath) => handleReadTextFile(targetPath));
  electron.ipcMain.handle("window:minimize", () => win?.minimize());
  electron.ipcMain.handle("window:toggle-maximize", () => {
    if (!win) return false;
    if (win.isMaximized()) {
      win.unmaximize();
      return false;
    }
    win.maximize();
    return true;
  });
  electron.ipcMain.handle("window:close", () => win?.close());
  electron.ipcMain.handle("window:is-maximized", () => win?.isMaximized() ?? false);
  createWindow();
  startEngine();
  electron.app.on("activate", () => {
    if (electron.BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});
electron.app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    stopEngine();
    electron.app.quit();
  }
});
electron.app.on("before-quit", () => {
  stopEngine();
});
