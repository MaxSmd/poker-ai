//! Minimal Slumbot HTTP client (`https://slumbot.com/slumbot/api/*`).
//!
//! JSON-over-POST with a session token; see [`crate::play::protocol`] for the
//! action-string format.  Transport errors are retried with backoff; protocol
//! errors (an `error_msg` in the response body) are surfaced to the caller.

use std::thread::sleep;
use std::time::Duration;

use serde::Deserialize;

/// A response from `/api/new_hand` or `/api/act`.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct HandResponse {
    #[serde(default)]
    pub old_action: Option<String>,
    #[serde(default)]
    pub action: Option<String>,
    #[serde(default)]
    pub client_pos: Option<u8>,
    #[serde(default)]
    pub hole_cards: Option<Vec<String>>,
    #[serde(default)]
    pub board: Option<Vec<String>>,
    /// Present exactly when the hand is over (chips, our perspective).
    #[serde(default)]
    pub winnings: Option<i64>,
    #[serde(default)]
    pub bot_hole_cards: Option<Vec<String>>,
    #[serde(default)]
    pub token: Option<String>,
    #[serde(default)]
    pub error_msg: Option<String>,
    #[serde(default)]
    pub session_num_hands: Option<u64>,
    #[serde(default)]
    pub session_total: Option<i64>,
    #[serde(default)]
    pub baseline_winnings: Option<i64>,
}

pub struct SlumbotClient {
    agent: ureq::Agent,
    base: String,
}

impl Default for SlumbotClient {
    fn default() -> Self {
        Self::new()
    }
}

impl SlumbotClient {
    pub fn new() -> Self {
        let agent = ureq::AgentBuilder::new()
            .timeout_read(Duration::from_secs(30))
            .timeout_write(Duration::from_secs(30))
            .build();
        Self { agent, base: "https://slumbot.com/slumbot/api".to_string() }
    }

    /// Point the client at a different server (tests / mirrors).
    pub fn with_base(mut self, base: impl Into<String>) -> Self {
        self.base = base.into();
        self
    }

    fn post(&self, endpoint: &str, body: serde_json::Value) -> Result<HandResponse, String> {
        let url = format!("{}/{}", self.base, endpoint);
        let mut last_err = String::new();
        for attempt in 0..4u32 {
            if attempt > 0 {
                sleep(Duration::from_secs(2u64 << attempt));
            }
            let sent = self
                .agent
                .post(&url)
                .set("Content-Type", "application/json")
                .send_string(&body.to_string());
            let text = match sent {
                Ok(resp) => resp.into_string().map_err(|e| e.to_string()),
                // 4xx/5xx still carry a JSON body with error_msg.
                Err(ureq::Error::Status(_code, resp)) => resp.into_string().map_err(|e| e.to_string()),
                Err(e) => {
                    last_err = format!("transport error: {e}");
                    continue; // retry
                }
            };
            let text = match text {
                Ok(t) => t,
                Err(e) => {
                    last_err = format!("read error: {e}");
                    continue;
                }
            };
            let parsed: HandResponse = match serde_json::from_str(&text) {
                Ok(p) => p,
                Err(e) => {
                    last_err = format!("bad JSON ({e}): {text}");
                    continue;
                }
            };
            if let Some(msg) = &parsed.error_msg {
                return Err(format!("server error: {msg}"));
            }
            return Ok(parsed);
        }
        Err(last_err)
    }

    /// Start a hand; `token` may be `None` on the very first request.
    pub fn new_hand(&self, token: Option<&str>) -> Result<HandResponse, String> {
        let body = match token {
            Some(t) => serde_json::json!({ "token": t }),
            None => serde_json::json!({}),
        };
        self.post("new_hand", body)
    }

    /// Send our incremental action (`"k" | "c" | "f" | "b<N>"`).
    pub fn act(&self, token: &str, incr: &str) -> Result<HandResponse, String> {
        self.post("act", serde_json::json!({ "token": token, "incr": incr }))
    }

    /// Log in a registered account; returns the session token.
    pub fn login(&self, username: &str, password: &str) -> Result<String, String> {
        let r = self.post("login", serde_json::json!({ "username": username, "password": password }))?;
        r.token.ok_or_else(|| "login response had no token".to_string())
    }
}
