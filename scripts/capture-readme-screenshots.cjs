const fs = require('node:fs');
const path = require('node:path');
const { app, BrowserWindow } = require('electron');

const repoRoot = process.cwd();
const outputDir = path.join(repoRoot, 'docs', 'screenshots');
const projectId = process.env.VF_SCREENSHOT_PROJECT_ID || '57c95cf7';
const baseUrl = process.env.VF_SCREENSHOT_BASE_URL || 'http://127.0.0.1:5173';

const targets = [
  { name: 'dashboard', route: '#/' },
  { name: 'projects', route: '#/projects' },
  { name: 'dataset', route: `#/projects/${projectId}/dataset` },
  { name: 'train-setup', route: `#/projects/${projectId}/train/setup` },
  { name: 'models', route: `#/projects/${projectId}/models` },
];

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function captureTarget(win, target) {
  const url = `${baseUrl}/${target.route}`;
  await win.loadURL(url);
  await wait(2500);
  const image = await win.webContents.capturePage();
  const outPath = path.join(outputDir, `${target.name}.png`);
  fs.writeFileSync(outPath, image.toPNG());
  process.stdout.write(`Captured ${outPath}\n`);
}

async function run() {
  fs.mkdirSync(outputDir, { recursive: true });
  const win = new BrowserWindow({
    width: 1660,
    height: 960,
    show: false,
    backgroundColor: '#071018',
    webPreferences: {
      offscreen: true,
      backgroundThrottling: false,
    },
  });
  for (const target of targets) {
    await captureTarget(win, target);
  }
  win.destroy();
}

app.whenReady().then(async () => {
  try {
    await run();
    app.quit();
  } catch (error) {
    console.error(error);
    app.exit(1);
  }
});

