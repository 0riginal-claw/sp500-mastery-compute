// index.js — Glitch container wrapper for the ZG validator.
//
// Spawns python3 zg_chain_node.py serve and exposes an HTTP healthcheck
// on $PORT (Glitch's required external port). The chain RPC stays on 9933
// internally; we proxy /health for Glitch's uptime probe.

const { spawn } = require("child_process");
const http = require("http");

const PORT = process.env.PORT || 3000;
const NODE_PORT = process.env.NODE_PORT || 9933;
const NODE_ID = process.env.PROJECT_NAME || `glitch-${Math.random().toString(36).slice(2,10)}`;
const PEERS = process.env.PEERS || "https://seed1.zgc.run,https://seed2.zgc.run";
const VALIDATOR_ADDR = process.env.ZG_VALIDATOR_ADDR || "";

const args = [
  "zg_chain_node.py", "serve",
  "--port", String(NODE_PORT),
  "--node-id", NODE_ID,
  "--peers", PEERS,
  "--sync-secs", "15",
  "--validator-reward", "1.0",
];
if (VALIDATOR_ADDR) {
  args.push("--validator-addr", VALIDATOR_ADDR);
}

console.log(`[glitch] booting ${NODE_ID} on :${NODE_PORT}, proxying healthcheck on :${PORT}`);

const proc = spawn("python3", args, { stdio: "inherit" });

proc.on("exit", (code) => {
  console.error(`[glitch] zg_chain_node exited with code ${code}`);
  process.exit(code === null ? 1 : code);
});

// Tiny HTTP proxy that exposes /health on $PORT so Glitch's idle-detection
// keeps the container alive when poked by UptimeRobot.
const server = http.createServer((req, res) => {
  if (req.url === "/health" || req.url === "/") {
    const proxyReq = http.request({
      host: "127.0.0.1",
      port: NODE_PORT,
      path: "/health",
      method: "POST",
      timeout: 4000,
    }, (proxyRes) => {
      res.writeHead(proxyRes.statusCode || 502, proxyRes.headers);
      proxyRes.pipe(res);
    });
    proxyReq.on("error", (e) => {
      res.writeHead(503, { "content-type": "application/json" });
      res.end(JSON.stringify({ ok: false, error: String(e) }));
    });
    proxyReq.end(JSON.stringify({}));
    return;
  }
  res.writeHead(404, { "content-type": "text/plain" });
  res.end("not found");
});

server.listen(PORT, () => {
  console.log(`[glitch] healthcheck proxy listening on :${PORT}`);
});
