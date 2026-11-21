//! Win-rate arithmetic shared by every match runner.
//!
//! A match is a stream of per-hand results in big blinds; what gets reported
//! is the mean in bb/100 and a 95% interval on it.  Every runner (Slumbot,
//! LBR) reports both a raw and a luck-adjusted figure, so the same three
//! running sums and two formulas were written out four times — once with the
//! `n < 2` guard and three times without.

/// A running win rate over per-hand results (bb).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WinRate {
    pub hands: u64,
    /// Net winnings so far (bb).
    pub net_bb: f64,
    sumsq_bb: f64,
}

impl WinRate {
    /// Add one hand's result (bb).
    pub fn record(&mut self, bb: f64) {
        self.hands += 1;
        self.net_bb += bb;
        self.sumsq_bb += bb * bb;
    }

    /// Mean result in bb/100 (`0` before any hand).
    pub fn bb100(&self) -> f64 {
        match self.hands {
            0 => 0.0,
            n => self.net_bb / n as f64 * 100.0,
        }
    }

    /// 95% confidence half-width on [`bb100`](Self::bb100), from the normal
    /// approximation (`0` with fewer than two hands, where no spread exists).
    pub fn ci95(&self) -> f64 {
        if self.hands < 2 {
            return 0.0;
        }
        let n = self.hands as f64;
        let mean = self.net_bb / n;
        let var = (self.sumsq_bb / n - mean * mean).max(0.0);
        1.96 * (var / n).sqrt() * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_and_interval_of_a_known_sample() {
        let mut w = WinRate::default();
        assert_eq!((w.bb100(), w.ci95()), (0.0, 0.0), "no hands is not a division by zero");
        for bb in [1.0, -1.0, 3.0, -3.0, 5.0] {
            w.record(bb);
        }
        // Mean 1 bb/hand; population variance (1+1+9+9+25)/5 − 1 = 8.
        assert!((w.bb100() - 100.0).abs() < 1e-9);
        assert!((w.ci95() - 1.96 * (8.0f64 / 5.0).sqrt() * 100.0).abs() < 1e-9);
        assert_eq!(w.hands, 5);
        assert_eq!(w.net_bb, 5.0);
    }
}
