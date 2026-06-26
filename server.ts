import express from "express";
import path from "path";
import { spawn, ChildProcess } from "child_process";
import { createServer as createViteServer } from "vite";
import { request as httpRequest } from "http";

const app = express();
const PORT = 3000;
let pythonProcess: ChildProcess | null = null;

// Start Python Flask app on port 5000
function startFlaskBackend() {
  console.log("[Node Server] Spawning Python Flask backend...");
  
  // Try running with python3 first, then fallback to python
  const pythonCmd = process.platform === "win32" ? "python" : "python3";
  const appPath = path.join(process.cwd(), "Music_Generation_AI", "app.py");

  pythonProcess = spawn(pythonCmd, [appPath], {
    stdio: "inherit",
    detached: false
  });

  pythonProcess.on("error", (err) => {
    console.error("[Node Server] Failed to start Python Flask process. Retrying with 'python' command...", err);
    if (pythonCmd === "python3") {
      pythonProcess = spawn("python", [appPath], {
        stdio: "inherit"
      });
    }
  });

  pythonProcess.on("exit", (code) => {
    console.log(`[Node Server] Python Flask backend exited with code ${code}`);
  });
}

// Safely terminate python process on shutdown
function cleanup() {
  if (pythonProcess) {
    console.log("[Node Server] Terminating Python Flask backend...");
    pythonProcess.kill();
    pythonProcess = null;
  }
}

process.on("exit", cleanup);
process.on("SIGINT", () => {
  cleanup();
  process.exit();
});
process.on("SIGTERM", () => {
  cleanup();
  process.exit();
});

async function startServer() {
  // Start the background python Flask app
  startFlaskBackend();

  // Express parser middlewares
  app.use(express.json({ limit: "50mb" }));
  app.use(express.urlencoded({ extended: true, limit: "50mb" }));

  // Lightweight API Proxy to Python Flask on port 5000
  app.all("/api/*", (req, res) => {
    const options = {
      hostname: "127.0.0.1",
      port: 5000,
      path: req.originalUrl,
      method: req.method,
      headers: req.headers,
    };

    const proxyReq = httpRequest(options, (proxyRes) => {
      res.writeHead(proxyRes.statusCode || 200, proxyRes.headers);
      proxyRes.pipe(res, { end: true });
    });

    req.pipe(proxyReq, { end: true });

    proxyReq.on("error", (err) => {
      console.error("[Node Proxy] Error forwarding request to Flask backend:", err.message);
      res.status(502).json({ 
        error: "Flask backend server not available", 
        details: err.message 
      });
    });
  });

  // Vite integration
  if (process.env.NODE_ENV !== "production") {
    console.log("[Node Server] Registering Vite development middleware...");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    console.log("[Node Server] Serving static files from production build directory...");
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`[Node Server] Running full-stack system at http://localhost:${PORT}`);
  });
}

startServer();
