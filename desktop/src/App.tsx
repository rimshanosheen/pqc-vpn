import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import "./App.css";

// Mirrors the Rust `ConnectionStatus` enum (src-tauri/src/lib.rs).
type ConnectionStatus = "disconnected" | "connecting" | "connected";

const STATUS_LABEL: Record<ConnectionStatus, string> = {
  disconnected: "Disconnected",
  connecting: "Connecting…",
  connected: "Connected",
};

// Mirrors `vpn::ClientConfig`.
type ClientConfig = { server_addr: string; verifying_key_hex: string };

// Mirrors `vpn::TickView`.
type TickView = {
  tick: number;
  in_force: string;
  raw_choice: string;
  probs: [number, number, number, number];
  gate_reason: string;
  action: string;
  outcome: string;
  state: number[];
  threat: number;
  anomalous: boolean;
  layer1_ready: boolean;
  layer2_active: boolean;
  layer3_active: boolean;
  session_id: string;
  psk_rotations: number;
};

// Mirrors `vpn::SessionEvent` (serde tag = "kind").
type SessionEvent =
  | { kind: "connected"; in_force: string; session_id: string; assigned_ip: string }
  | ({ kind: "tick" } & TickView)
  | { kind: "error"; message: string }
  | { kind: "stopped" };

// Mirrors the Rust `DeviceSnapshot` struct.
type DeviceSnapshot = {
  cpu_load: number;
  ram_available_fraction: number;
  latency_ms: number | null;
  upload_rate_bytes_per_sec: number;
  connection_type: string;
};

const ACTIONS = ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024", "rekey-now"];
const STATE_NAMES = ["CPU", "RAM free", "Latency", "Upload", "Link", "Key age", "Threat"];

const percent = (fraction: number) => `${Math.round(fraction * 100)}%`;

function App() {
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const [pending, setPending] = useState(false);
  const [config, setConfig] = useState<ClientConfig | null>(null);
  const [connectedInfo, setConnectedInfo] = useState<string | null>(null);
  const [tick, setTick] = useState<TickView | null>(null);
  const [log, setLog] = useState<string[]>([]);
  const [device, setDevice] = useState<DeviceSnapshot | null>(null);
  const [deviceLoading, setDeviceLoading] = useState(false);

  const addLog = (line: string) =>
    setLog((l) => [`${new Date().toLocaleTimeString()}  ${line}`, ...l].slice(0, 12));

  useEffect(() => {
    invoke<ConnectionStatus>("connection_status").then(setStatus);
    invoke<ClientConfig>("default_config").then(setConfig);

    const unStatus = listen<ConnectionStatus>("vpn-status", (e) => setStatus(e.payload));
    const unEvent = listen<SessionEvent>("vpn-event", (e) => {
      const ev = e.payload;
      switch (ev.kind) {
        case "connected":
          setConnectedInfo(`${ev.in_force} · tunnel IP ${ev.assigned_ip}`);
          addLog(`connected at ${ev.in_force} (agent's choice), session ${ev.session_id.slice(0, 8)}…`);
          break;
        case "tick":
          setTick(ev);
          // Only log ticks where something happened, or the log is noise.
          if (ev.action !== "stay" || ev.anomalous) {
            addLog(`tick ${ev.tick}: ${ev.action} → ${ev.outcome}`);
          }
          break;
        case "error":
          addLog(`error: ${ev.message}`);
          break;
        case "stopped":
          addLog("session stopped");
          setConnectedInfo(null);
          break;
      }
    });
    return () => {
      unStatus.then((f) => f());
      unEvent.then((f) => f());
    };
  }, []);

  async function toggleConnection() {
    setPending(true);
    try {
      if (status === "disconnected") {
        setTick(null);
        setStatus(await invoke<ConnectionStatus>("connect", { config }));
      } else {
        setStatus(await invoke<ConnectionStatus>("disconnect"));
      }
    } catch (err) {
      addLog(`error: ${err}`);
    } finally {
      setPending(false);
    }
  }

  async function refreshDevice() {
    setDeviceLoading(true);
    try {
      setDevice(await invoke<DeviceSnapshot>("device_snapshot"));
    } finally {
      setDeviceLoading(false);
    }
  }

  return (
    <main className="container">
      <h1>PQC VPN</h1>

      <div className={`status status--${status}`}>
        <span className="status__dot" />
        {STATUS_LABEL[status]}
        {connectedInfo && status === "connected" ? ` · ${connectedInfo}` : ""}
      </div>

      {config && status === "disconnected" && (
        <section className="device-panel">
          <div className="device-panel__header">
            <h2>Server</h2>
          </div>
          <label className="field">
            Handshake address
            <input
              value={config.server_addr}
              onChange={(e) => setConfig({ ...config, server_addr: e.target.value })}
            />
          </label>
          <label className="field">
            Pinned server key (ML-DSA-65, hex)
            <textarea
              rows={2}
              value={config.verifying_key_hex}
              onChange={(e) => setConfig({ ...config, verifying_key_hex: e.target.value })}
            />
          </label>
        </section>
      )}

      <button className="connect-button" onClick={toggleConnection} disabled={pending || !config}>
        {status === "disconnected" ? "Connect" : "Disconnect"}
      </button>

      {/* Week 8 vertical slice: what the agent saw and did on the last
          5-second tick. The WireGuard interface itself is Week 9; the PSK
          rotations counted here are the ones the server installed. */}
      {tick && (
        <section className="device-panel">
          <div className="device-panel__header">
            <h2>Agent · tick {tick.tick}</h2>
            <span className="pill">{tick.in_force}</span>
          </div>

          <dl className="device-grid">
            <dt>Threat score</dt>
            <dd>
              {tick.threat.toFixed(3)}
              {tick.anomalous ? " ⚠" : ""}
            </dd>
            <dt>Anomaly layers</dt>
            <dd>
              L1 {tick.layer1_ready ? "ready" : "warming up"} · L2 {tick.layer2_active ? "on" : "off"} · L3{" "}
              {tick.layer3_active ? "on" : "off"}
            </dd>
            <dt>Policy wants</dt>
            <dd>{tick.raw_choice}</dd>
            <dt>Gate</dt>
            <dd className="small">{tick.gate_reason}</dd>
            <dt>Action</dt>
            <dd>{tick.outcome === "unchanged" ? "stay" : tick.outcome}</dd>
            <dt>PSK rotations</dt>
            <dd>{tick.psk_rotations}</dd>
          </dl>

          <div className="bars">
            {ACTIONS.map((name, i) => (
              <div key={name} className="bar">
                <span>{name}</span>
                <div className="bar__track">
                  <div className="bar__fill" style={{ width: percent(tick.probs[i]) }} />
                </div>
                <span>{percent(tick.probs[i])}</span>
              </div>
            ))}
          </div>

          <div className="state-row">
            {STATE_NAMES.map((name, i) => (
              <span key={name} title={name}>
                {name} {tick.state[i].toFixed(2)}
              </span>
            ))}
          </div>
        </section>
      )}

      {log.length > 0 && (
        <section className="device-panel">
          <div className="device-panel__header">
            <h2>Events</h2>
          </div>
          <ul className="log">
            {log.map((l, i) => (
              <li key={i}>{l}</li>
            ))}
          </ul>
        </section>
      )}

      <section className="device-panel">
        <div className="device-panel__header">
          <h2>Device</h2>
          <button onClick={refreshDevice} disabled={deviceLoading}>
            {deviceLoading ? "Reading…" : "Refresh"}
          </button>
        </div>

        {device ? (
          <dl className="device-grid">
            <dt>CPU load</dt>
            <dd>{percent(device.cpu_load)}</dd>

            <dt>RAM available</dt>
            <dd>{percent(device.ram_available_fraction)}</dd>

            <dt>Latency</dt>
            <dd>{device.latency_ms === null ? "unreachable" : `${device.latency_ms.toFixed(0)} ms`}</dd>

            <dt>Upload rate</dt>
            <dd>{(device.upload_rate_bytes_per_sec / 1000).toFixed(1)} KB/s</dd>

            <dt>Connection</dt>
            <dd>{device.connection_type}</dd>
          </dl>
        ) : (
          <p className="device-panel__empty">Not read yet — click Refresh for live numbers.</p>
        )}
      </section>
    </main>
  );
}

export default App;
