//! Discounted Counterfactual Regret Minimization (DCFR).
//!
//! Brown & Sandholm, AAAI 2019.
//! Discount: d(t) = t^alpha / (t^alpha + 1), alpha ≈ 1.5.

pub const ALPHA: f64 = 1.5;

/// Compute the DCFR discount factor for iteration `t`.
pub fn discount(t: u64) -> f64 {
    let t = t as f64;
    t.powf(ALPHA) / (t.powf(ALPHA) + 1.0)
}

/// Apply the DCFR update to a regret slice for one info set.
pub fn update_regrets(regrets: &mut [f32], new_regrets: &[f32], t: u64) {
    let d = discount(t) as f32;
    for (r, &nr) in regrets.iter_mut().zip(new_regrets.iter()) {
        *r = *r * d + nr;
    }
}
