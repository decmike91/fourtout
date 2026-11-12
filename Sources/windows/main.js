const { app, BrowserWindow, Menu, shell } = require("electron");
const path = require("path");

if (!app.requestSingleInstanceLock()) app.quit();

function createWindow() {
  const win = new BrowserWindow({
    width: 1280, height: 820, minWidth: 420, minHeight: 500,
    title: "Trait Fin", icon: path.join(__dirname, "icon.png"),
    backgroundColor: "#E8ECF1", autoHideMenuBar: true,
    webPreferences: { contextIsolation: true, sandbox: true }
  });
  Menu.setApplicationMenu(null);
  win.loadFile(path.join(__dirname, "index.html"));
  // Liens externes dans le navigateur par défaut
  win.webContents.setWindowOpenHandler(({ url }) => { if (/^https?:/.test(url)) shell.openExternal(url); return { action: "deny" }; });
  // Enregistrement : boîte de dialogue « Enregistrer sous » native
  win.webContents.session.on("will-download", (e, item) => {
    item.setSaveDialogOptions({ title: "Enregistrer le SVG", filters: [{ name: "Image SVG", extensions: ["svg"] }] });
  });
}

app.on("second-instance", () => { const w = BrowserWindow.getAllWindows()[0]; if (w) { if (w.isMinimized()) w.restore(); w.focus(); } });
app.whenReady().then(createWindow);
app.on("window-all-closed", () => app.quit());
