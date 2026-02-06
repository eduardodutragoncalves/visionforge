"use strict";
const electron = require("electron");
const path = require("node:path");
const node_os = require("node:os");
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
let win = null;
async function handleOpenDirectory() {
  const { canceled, filePaths } = await electron.dialog.showOpenDialog({
    properties: ["openDirectory", "createDirectory"]
  });
  if (canceled) {
    return null;
  }
  return filePaths[0];
}
function createWindow() {
  win = new electron.BrowserWindow({
    icon: path.join(PUBLIC, "electron-vite.svg"),
    width: 1200,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, "preload.js")
    },
    titleBarStyle: "hidden",
    titleBarOverlay: {
      color: "rgba(0,0,0,0)",
      symbolColor: "#ffffff"
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
  if (VITE_DEV_SERVER_URL) {
    win.loadURL(VITE_DEV_SERVER_URL);
  } else {
    win.loadFile(path.join(DIST, "index.html"));
  }
}
electron.app.whenReady().then(() => {
  electron.ipcMain.handle("dialog:openDirectory", handleOpenDirectory);
  createWindow();
  electron.app.on("activate", () => {
    if (electron.BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});
electron.app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    electron.app.quit();
  }
});
