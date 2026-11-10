import { spawn } from "node:child_process";
import { isAbsolute, resolve } from "node:path";

const npm = process.platform === "win32" ? "npm.cmd" : "npm";
const configuredPython = process.env.MODEL_PYTHON;
const python = configuredPython
  ? isAbsolute(configuredPython) || !/[\\/]/.test(configuredPython)
    ? configuredPython
    : resolve(configuredPython)
  : (process.platform === "win32" ? "python" : "python3");
const services = [
  ...(process.env.MODEL_SERVICE_AUTOSTART === "false"
    ? []
    : [["model service", spawn(python, ["app.py"], {
      cwd: resolve("model-service"),
      env: { ...process.env, MODEL_HOST: "127.0.0.1", MODEL_PORT: "5000" },
      stdio: "inherit",
    })]]),
  ["backend", spawn(npm, ["--prefix", "backend", "run", "dev"], { stdio: "inherit" })],
  ["frontend", spawn(npm, ["--prefix", "frontend", "run", "dev"], { stdio: "inherit" })],
];

let stopping = false;

function stopAll(exitCode = 0) {
  if (stopping) return;
  stopping = true;
  for (const [, child] of services) {
    if (child.exitCode === null && !child.killed) child.kill("SIGTERM");
  }
  process.exitCode = exitCode;
}

for (const [name, child] of services) {
  child.on("error", (error) => {
    console.error(`Could not start the ${name}: ${error.message}`);
    stopAll(1);
  });
  child.on("exit", (code, signal) => {
    if (!stopping) {
      console.error(`${name} stopped${signal ? ` (${signal})` : ` with exit code ${code}`}.`);
      stopAll(code || 1);
    }
  });
}

process.on("SIGINT", () => stopAll(0));
process.on("SIGTERM", () => stopAll(0));
