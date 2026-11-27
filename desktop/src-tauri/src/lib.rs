//! Tauri desktop shell.
//!
//! Week 4: window + system tray. Week 5: `device_state` implements
//! `core::state::DeviceState` with `sysinfo`. Week 8 (the first vertical
//! slice): `traffic` implements `core::state::TrafficSignals` (Layer 2's
//! live inputs), and Connect now runs the real loop in `vpn`: live readings
//! → anomaly layers → RL agent → real handshakes against the server, every
//! 5 s, each tick pushed to the frontend as a `vpn-event`.
//!
//! Still not real: the local WireGuard interface (`vpn::PendingTunnel`,
//! replaced by a `wg`-backed `TunnelHandle` in Week 9).

pub mod device_state;
pub mod traffic;
pub mod vpn;

use std::sync::Mutex;

use serde::Serialize;
use tauri::menu::{Menu, MenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::{AppHandle, Emitter, Manager, State};
use vpn_core::state::DeviceState;

use device_state::SysinfoDeviceState;
use vpn::{ClientConfig, Running, SessionEvent};

/// Connection state the frontend can read and drive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ConnectionStatus {
    Disconnected,
    Connecting,
    Connected,
}

/// App-wide state, managed by Tauri and shared across command invocations.
struct AppState {
    status: Mutex<ConnectionStatus>,
    device: SysinfoDeviceState,
    session: Mutex<Option<Running>>,
}

/// A `DeviceState` reading, shaped for the frontend. Plain fields rather
/// than reusing `vpn_core::state::ConnectionType` directly — `core` does not
/// (and should not) depend on `serde`, so the enum has no `Serialize` impl;
/// its `Debug` output ("Wired" / "WifiOrUnknown" / "Cellular") is sent
/// as-is instead.
#[derive(Debug, Clone, Serialize)]
struct DeviceSnapshot {
    cpu_load: f32,
    ram_available_fraction: f32,
    /// `None` when the latency probe's target was unreachable.
    latency_ms: Option<f64>,
    upload_rate_bytes_per_sec: f64,
    connection_type: String,
}

/// Reads the five `DeviceState` values live from this machine, on demand.
#[tauri::command]
fn device_snapshot(state: State<AppState>) -> DeviceSnapshot {
    let device = &state.device;

    // cpu_load blocks ~200ms (sysinfo's minimum sample interval) and
    // latency blocks up to ~1s (the ping timeout).
    let cpu_load = device.cpu_load();
    let ram_available_fraction = device.ram_available_fraction();
    let latency_ms = device.latency().map(|d| d.as_secs_f64() * 1000.0);
    let upload_rate_bytes_per_sec = device.upload_rate_bytes_per_sec();
    let connection_type = format!("{:?}", device.connection_type());

    DeviceSnapshot {
        cpu_load,
        ram_available_fraction,
        latency_ms,
        upload_rate_bytes_per_sec,
        connection_type,
    }
}

fn set_status(app: &AppHandle, status: ConnectionStatus) {
    *app.state::<AppState>().status.lock().expect("status mutex poisoned") = status;
    let _ = app.emit("vpn-status", status);
}

/// The server address and pinned key the app ships with, for the
/// frontend's connection form.
#[tauri::command]
fn default_config() -> ClientConfig {
    ClientConfig::default()
}

/// Starts a session: the handshake and the 5-second loop run on a
/// background thread, so this returns `Connecting` straight away. Progress
/// arrives as `vpn-event` (connected / tick / error / stopped) and
/// `vpn-status` events.
#[tauri::command]
fn connect(
    app: AppHandle,
    state: State<AppState>,
    config: ClientConfig,
) -> Result<ConnectionStatus, String> {
    let mut slot = state.session.lock().expect("session mutex poisoned");
    if slot.is_some() {
        return Err("already connected; disconnect first".into());
    }

    let events = app.clone();
    let running = vpn::start(config, move |event| {
        match &event {
            SessionEvent::Connected { .. } => set_status(&events, ConnectionStatus::Connected),
            SessionEvent::Error { .. } | SessionEvent::Stopped => {
                set_status(&events, ConnectionStatus::Disconnected);
            }
            SessionEvent::Tick(_) => {}
        }
        let _ = events.emit("vpn-event", event);
    })?;

    *slot = Some(running);
    set_status(&app, ConnectionStatus::Connecting);
    Ok(ConnectionStatus::Connecting)
}

/// Stops the session loop and tears the tunnel down. The loop finishes its
/// current tick first, so this waits on a helper thread rather than
/// blocking the UI.
#[tauri::command]
fn disconnect(app: AppHandle, state: State<AppState>) -> ConnectionStatus {
    if let Some(running) = state.session.lock().expect("session mutex poisoned").take() {
        std::thread::spawn(move || running.stop());
    }
    set_status(&app, ConnectionStatus::Disconnected);
    ConnectionStatus::Disconnected
}

/// Lets the frontend read the current status without changing it — used on
/// startup and whenever the window regains focus.
#[tauri::command]
fn connection_status(state: State<AppState>) -> ConnectionStatus {
    *state.status.lock().expect("status mutex poisoned")
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(AppState {
            status: Mutex::new(ConnectionStatus::Disconnected),
            device: SysinfoDeviceState::new(),
            session: Mutex::new(None),
        })
        .setup(|app| {
            // System tray: an icon plus a minimal Quit menu.
            let quit_item = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;

            let tray_menu = Menu::with_items(app, &[&quit_item])?;

            TrayIconBuilder::new()
                .icon(
                    app.default_window_icon()
                        .expect("the app icon is bundled at build time")
                        .clone(),
                )
                .menu(&tray_menu)
                .show_menu_on_left_click(true)
                .on_menu_event(|app, event| {
                    if event.id.as_ref() == "quit" {
                        app.exit(0);
                    }
                })
                .build(app)?;

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            connect,
            disconnect,
            connection_status,
            device_snapshot,
            default_config
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
