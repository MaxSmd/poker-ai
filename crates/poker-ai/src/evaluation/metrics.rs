//! Node-per-second counter and convergence tracking.

use std::time::Instant;

pub struct NpsCounter {
    start: Instant,
    nodes: u64,
}

impl Default for NpsCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl NpsCounter {
    pub fn new() -> Self {
        Self { start: Instant::now(), nodes: 0 }
    }

    pub fn increment(&mut self, count: u64) {
        self.nodes += count;
    }

    pub fn nps(&self) -> f64 {
        let elapsed = self.start.elapsed().as_secs_f64();
        if elapsed > 0.0 { self.nodes as f64 / elapsed } else { 0.0 }
    }
}
