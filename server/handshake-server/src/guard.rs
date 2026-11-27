//! Replay protection and handshake rate limiting (`PROTOCOL.md` §7, Week 7).
//!
//! Everything here is server-local — no wire-format change. Three independent
//! checks, all cheap and all run *before* the server does any ML-KEM / ML-DSA
//! work:
//!
//! - [`NonceCache`] — every `client_nonce` (from a `ClientHello` or the hello
//!   inside a `RekeyRequest`) is remembered for `replay_window`. A second
//!   message carrying the same nonce inside that window is rejected. A replayed
//!   hello could never *complete* (the replayer lacks the client's ephemeral
//!   secrets, so it cannot produce a valid `ClientFinish`), but without this
//!   cache each replay would still cost the server an ML-DSA-65 signature.
//! - [`RateLimiter`] — a token bucket per source IP: `burst` handshakes
//!   immediately, then refilled at `per_minute`. Over the limit →
//!   `Error(0x07 RATE_LIMITED)` and the connection is closed.
//! - a global cap on concurrently open connections, because the server is
//!   blocking thread-per-connection with a 10 s socket timeout: without a cap,
//!   a client that opens sockets and goes silent ties up one thread each.
//!
//! Both tables are bounded (`MAX_TRACKED_*`); when one is full even after
//! pruning expired entries, the check fails closed (rejects) rather than
//! growing without limit.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// `PROTOCOL.md` §7: "reject duplicates within a 2-minute window".
pub const DEFAULT_REPLAY_WINDOW: Duration = Duration::from_secs(120);
/// Handshakes a single source IP may start back-to-back. Generous on purpose:
/// several demo laptops behind one NAT (a campus Wi-Fi) share one public IP.
pub const DEFAULT_RATE_BURST: u32 = 20;
/// Sustained handshakes per minute per source IP once the burst is spent.
pub const DEFAULT_RATE_PER_MINUTE: u32 = 30;
/// Concurrently open handshake connections, server-wide.
pub const DEFAULT_MAX_CONCURRENT: usize = 64;

/// Upper bound on remembered nonces. At the default rate limit a single IP
/// can add at most ~80 per window, so this is only reached under a wide,
/// many-IP flood — then new handshakes are refused until entries expire.
pub const MAX_TRACKED_NONCES: usize = 100_000;
/// Upper bound on per-IP buckets.
pub const MAX_TRACKED_IPS: usize = 10_000;

/// Why the guard refused a connection or message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rejection {
    /// Per-IP token bucket empty, or the IP table is full.
    RateLimited,
    /// Too many connections open at once.
    TooManyConnections,
    /// This `client_nonce` was already seen inside the replay window.
    Replay,
    /// The nonce table is full of unexpired entries.
    NonceCacheFull,
}

impl Rejection {
    /// The `PROTOCOL.md` §4.7 error code sent to the client.
    ///
    /// Replays map to `0x06 AUTH_FAILED`: the frozen v1 code table has no
    /// dedicated replay code, and from the protocol's point of view a reused
    /// nonce is a handshake that cannot be authenticated as fresh. A real
    /// client never sees it — its nonces are 32 random bytes. Everything
    /// load-related is `0x07 RATE_LIMITED`.
    pub fn protocol_code(self) -> u8 {
        match self {
            Rejection::Replay => 0x06,
            Rejection::RateLimited | Rejection::TooManyConnections | Rejection::NonceCacheFull => 0x07,
        }
    }

    pub fn message(self) -> &'static str {
        match self {
            Rejection::RateLimited => "rate limited: too many handshakes from this address",
            Rejection::TooManyConnections => "rate limited: server busy",
            Rejection::Replay => "client nonce already used",
            Rejection::NonceCacheFull => "rate limited: server busy",
        }
    }
}

impl std::fmt::Display for Rejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

// ---------------------------------------------------------------------------
// Replay cache
// ---------------------------------------------------------------------------

/// Remembers recently seen `client_nonce` values.
pub struct NonceCache {
    window: Duration,
    capacity: usize,
    seen: Mutex<HashMap<[u8; 32], Instant>>,
}

impl NonceCache {
    pub fn new(window: Duration, capacity: usize) -> Self {
        Self { window, capacity, seen: Mutex::new(HashMap::new()) }
    }

    /// Record `nonce` as seen at `now`. `Err(Replay)` if it was already seen
    /// within the window.
    pub fn check_and_insert(&self, nonce: &[u8; 32], now: Instant) -> Result<(), Rejection> {
        let mut seen = self.seen.lock().unwrap();
        if let Some(&at) = seen.get(nonce) {
            if now.saturating_duration_since(at) < self.window {
                return Err(Rejection::Replay);
            }
        }
        if seen.len() >= self.capacity && !seen.contains_key(nonce) {
            let window = self.window;
            seen.retain(|_, at| now.saturating_duration_since(*at) < window);
            if seen.len() >= self.capacity {
                return Err(Rejection::NonceCacheFull);
            }
        }
        seen.insert(*nonce, now);
        Ok(())
    }

    /// Drop expired entries. Called periodically by the housekeeping thread.
    pub fn prune(&self, now: Instant) {
        let window = self.window;
        self.seen.lock().unwrap().retain(|_, at| now.saturating_duration_since(*at) < window);
    }

    pub fn len(&self) -> usize {
        self.seen.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ---------------------------------------------------------------------------
// Per-IP token bucket
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct Bucket {
    tokens: f64,
    last: Instant,
}

/// Token-bucket rate limiter keyed by source IP.
pub struct RateLimiter {
    burst: f64,
    per_sec: f64,
    capacity: usize,
    buckets: Mutex<HashMap<IpAddr, Bucket>>,
}

impl RateLimiter {
    /// `per_minute == 0` means no refill: each IP gets `burst` handshakes total.
    pub fn new(burst: u32, per_minute: u32, capacity: usize) -> Self {
        Self {
            burst: burst as f64,
            per_sec: per_minute as f64 / 60.0,
            capacity,
            buckets: Mutex::new(HashMap::new()),
        }
    }

    fn refilled(&self, b: Bucket, now: Instant) -> f64 {
        let dt = now.saturating_duration_since(b.last).as_secs_f64();
        (b.tokens + dt * self.per_sec).min(self.burst)
    }

    /// Spend one token for `ip`. `Err(RateLimited)` if it has none left.
    pub fn check(&self, ip: IpAddr, now: Instant) -> Result<(), Rejection> {
        let mut buckets = self.buckets.lock().unwrap();
        if !buckets.contains_key(&ip) && buckets.len() >= self.capacity {
            self.prune_locked(&mut buckets, now);
            if buckets.len() >= self.capacity {
                return Err(Rejection::RateLimited);
            }
        }
        let b = buckets.entry(ip).or_insert(Bucket { tokens: self.burst, last: now });
        let tokens = self.refilled(*b, now);
        if tokens < 1.0 {
            // Keep `last` unchanged so the partial refill keeps accruing.
            return Err(Rejection::RateLimited);
        }
        *b = Bucket { tokens: tokens - 1.0, last: now };
        Ok(())
    }

    /// Forget IPs whose bucket has refilled completely — they are
    /// indistinguishable from an IP never seen.
    pub fn prune(&self, now: Instant) {
        let mut buckets = self.buckets.lock().unwrap();
        self.prune_locked(&mut buckets, now);
    }

    fn prune_locked(&self, buckets: &mut HashMap<IpAddr, Bucket>, now: Instant) {
        buckets.retain(|_, b| self.refilled(*b, now) < self.burst);
    }

    pub fn tracked_ips(&self) -> usize {
        self.buckets.lock().unwrap().len()
    }
}

// ---------------------------------------------------------------------------
// The guard as a whole
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct GuardConfig {
    pub replay_window: Duration,
    pub rate_burst: u32,
    pub rate_per_minute: u32,
    pub max_concurrent: usize,
}

impl Default for GuardConfig {
    fn default() -> Self {
        Self {
            replay_window: DEFAULT_REPLAY_WINDOW,
            rate_burst: DEFAULT_RATE_BURST,
            rate_per_minute: DEFAULT_RATE_PER_MINUTE,
            max_concurrent: DEFAULT_MAX_CONCURRENT,
        }
    }
}

/// Replay cache + rate limiter + concurrency cap, shared by every connection.
pub struct Guard {
    pub config: GuardConfig,
    pub nonces: NonceCache,
    pub limiter: RateLimiter,
    active: Arc<AtomicUsize>,
}

/// Held for the lifetime of one admitted connection; releases its
/// concurrency slot on drop (including on panic / early return).
pub struct Permit {
    active: Arc<AtomicUsize>,
}

impl Drop for Permit {
    fn drop(&mut self) {
        self.active.fetch_sub(1, Ordering::AcqRel);
    }
}

impl Guard {
    pub fn new(config: GuardConfig) -> Self {
        Self {
            nonces: NonceCache::new(config.replay_window, MAX_TRACKED_NONCES),
            limiter: RateLimiter::new(config.rate_burst, config.rate_per_minute, MAX_TRACKED_IPS),
            active: Arc::new(AtomicUsize::new(0)),
            config,
        }
    }

    /// Admission check for a freshly accepted connection from `ip`: a
    /// concurrency slot first (so a refused connection does not spend the
    /// IP's token), then the per-IP rate limit.
    pub fn admit(&self, ip: IpAddr, now: Instant) -> Result<Permit, Rejection> {
        let prev = self.active.fetch_add(1, Ordering::AcqRel);
        let permit = Permit { active: Arc::clone(&self.active) };
        if prev >= self.config.max_concurrent {
            return Err(Rejection::TooManyConnections); // permit drops → slot released
        }
        self.limiter.check(ip, now)?;
        Ok(permit)
    }

    /// Replay check for the `client_nonce` of message 1.
    pub fn check_nonce(&self, nonce: &[u8; 32], now: Instant) -> Result<(), Rejection> {
        self.nonces.check_and_insert(nonce, now)
    }

    pub fn active_connections(&self) -> usize {
        self.active.load(Ordering::Acquire)
    }

    pub fn prune(&self, now: Instant) {
        self.nonces.prune(now);
        self.limiter.prune(now);
    }
}

impl Default for Guard {
    fn default() -> Self {
        Self::new(GuardConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    fn ip(n: u8) -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(203, 0, 113, n))
    }

    #[test]
    fn nonce_seen_twice_in_window_is_a_replay() {
        let c = NonceCache::new(Duration::from_secs(120), 100);
        let t = Instant::now();
        assert_eq!(c.check_and_insert(&[1; 32], t), Ok(()));
        assert_eq!(c.check_and_insert(&[1; 32], t + Duration::from_secs(119)), Err(Rejection::Replay));
        assert_eq!(c.check_and_insert(&[2; 32], t), Ok(()), "a different nonce is fine");
    }

    #[test]
    fn nonce_is_accepted_again_after_the_window() {
        let c = NonceCache::new(Duration::from_secs(120), 100);
        let t = Instant::now();
        c.check_and_insert(&[1; 32], t).unwrap();
        assert_eq!(c.check_and_insert(&[1; 32], t + Duration::from_secs(120)), Ok(()));
    }

    #[test]
    fn prune_drops_only_expired_nonces() {
        let c = NonceCache::new(Duration::from_secs(10), 100);
        let t = Instant::now();
        c.check_and_insert(&[1; 32], t).unwrap();
        c.check_and_insert(&[2; 32], t + Duration::from_secs(8)).unwrap();
        c.prune(t + Duration::from_secs(12));
        assert_eq!(c.len(), 1);
        assert_eq!(c.check_and_insert(&[2; 32], t + Duration::from_secs(12)), Err(Rejection::Replay));
    }

    #[test]
    fn full_nonce_cache_evicts_expired_then_fails_closed() {
        let c = NonceCache::new(Duration::from_secs(10), 2);
        let t = Instant::now();
        c.check_and_insert(&[1; 32], t).unwrap();
        c.check_and_insert(&[2; 32], t).unwrap();
        // full, nothing expired → refuse rather than grow
        assert_eq!(c.check_and_insert(&[3; 32], t + Duration::from_secs(1)), Err(Rejection::NonceCacheFull));
        // once the old ones expire, there is room again
        assert_eq!(c.check_and_insert(&[3; 32], t + Duration::from_secs(11)), Ok(()));
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn bucket_allows_burst_then_limits() {
        let r = RateLimiter::new(3, 60, 100);
        let t = Instant::now();
        for _ in 0..3 {
            assert_eq!(r.check(ip(1), t), Ok(()));
        }
        assert_eq!(r.check(ip(1), t), Err(Rejection::RateLimited));
        // another IP has its own bucket
        assert_eq!(r.check(ip(2), t), Ok(()));
    }

    #[test]
    fn bucket_refills_at_the_configured_rate() {
        let r = RateLimiter::new(2, 60, 100); // 1 token per second
        let t = Instant::now();
        r.check(ip(1), t).unwrap();
        r.check(ip(1), t).unwrap();
        assert!(r.check(ip(1), t + Duration::from_millis(500)).is_err());
        // the refused attempt above must not reset the refill clock
        assert_eq!(r.check(ip(1), t + Duration::from_millis(1000)), Ok(()));
        assert!(r.check(ip(1), t + Duration::from_millis(1000)).is_err());
        // refill never exceeds the burst
        let later = t + Duration::from_secs(3600);
        assert!(r.check(ip(1), later).is_ok());
        assert!(r.check(ip(1), later).is_ok());
        assert!(r.check(ip(1), later).is_err());
    }

    #[test]
    fn zero_refill_means_burst_is_a_hard_total() {
        let r = RateLimiter::new(1, 0, 100);
        let t = Instant::now();
        r.check(ip(1), t).unwrap();
        assert!(r.check(ip(1), t + Duration::from_secs(10_000)).is_err());
    }

    #[test]
    fn prune_forgets_fully_refilled_ips_only() {
        let r = RateLimiter::new(2, 60, 100);
        let t = Instant::now();
        r.check(ip(1), t).unwrap();
        r.check(ip(2), t + Duration::from_secs(5)).unwrap();
        r.prune(t + Duration::from_millis(5500)); // ip1 refilled, ip2 not yet
        assert_eq!(r.tracked_ips(), 1);
    }

    #[test]
    fn full_ip_table_fails_closed_for_new_ips_only() {
        let r = RateLimiter::new(5, 0, 2);
        let t = Instant::now();
        r.check(ip(1), t).unwrap();
        r.check(ip(2), t).unwrap();
        assert_eq!(r.check(ip(3), t), Err(Rejection::RateLimited));
        assert_eq!(r.check(ip(1), t), Ok(()), "already-tracked IPs keep working");
    }

    #[test]
    fn concurrency_slots_are_released_on_drop() {
        let g = Guard::new(GuardConfig { max_concurrent: 2, ..GuardConfig::default() });
        let t = Instant::now();
        let a = g.admit(ip(1), t).unwrap();
        let _b = g.admit(ip(2), t).unwrap();
        assert_eq!(g.admit(ip(3), t).err(), Some(Rejection::TooManyConnections));
        assert_eq!(g.active_connections(), 2, "a refused admit must not leak a slot");
        drop(a);
        assert!(g.admit(ip(3), t).is_ok());
    }

    #[test]
    fn busy_refusal_does_not_spend_the_ips_token() {
        let g = Guard::new(GuardConfig { max_concurrent: 1, rate_burst: 1, rate_per_minute: 0, ..GuardConfig::default() });
        let t = Instant::now();
        let held = g.admit(ip(1), t).unwrap();
        assert_eq!(g.admit(ip(2), t).err(), Some(Rejection::TooManyConnections));
        drop(held);
        assert!(g.admit(ip(2), t).is_ok(), "ip2's single token was still unspent");
    }

    #[test]
    fn protocol_codes() {
        assert_eq!(Rejection::Replay.protocol_code(), 0x06);
        assert_eq!(Rejection::RateLimited.protocol_code(), 0x07);
        assert_eq!(Rejection::TooManyConnections.protocol_code(), 0x07);
        assert_eq!(Rejection::NonceCacheFull.protocol_code(), 0x07);
    }
}
