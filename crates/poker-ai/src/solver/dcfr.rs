//! Discounted Counterfactual Regret Minimization (DCFR) discount schedule.
//!
//! Brown & Sandholm, AAAI 2019.  We use the full three-parameter
//! form `(α, β, γ) = (1.5, 0, 2)`, not an α-only discount:
//!
//! * `α = 1.5` — discounts accumulated **positive** regret, suppressing the
//!   noise of early iterations.
//! * `β = 0`   — applies a constant 0.5 weight to accumulated **negative**
//!   regret, letting actions that early noise made look bad recover quickly.
//! * `γ = 2`   — weights the **strategy-sum** accumulation toward later
//!   iterations.  This is the parameter that most directly improves the
//!   deployed blueprint, because the blueprint is the time-averaged strategy,
//!   not the last iterate.
//!
//! Each iteration `t` (1-indexed), before adding the new instantaneous regret,
//! the running positive/negative regret is multiplied by `t^x / (t^x + 1)`.
//! The strategy-sum contribution on iteration `t` carries weight `∝ t^γ`, which
//! is equivalent to DCFR's `(t/(t+1))^γ` running discount up to normalization.

/// DCFR discount parameters.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct Discount {
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
}

impl Discount {
    /// The recommended `(α, β, γ) = (1.5, 0, 2)`.
    pub const RECOMMENDED: Discount = Discount { alpha: 1.5, beta: 0.0, gamma: 2.0 };

    /// Linear CFR (Brown & Sandholm 2019): `(α, β, γ) = (1, 1, 1)` — iteration
    /// `t` weighted by `t` everywhere.  Converges somewhat slower than
    /// [`RECOMMENDED`](Self::RECOMMENDED) per iteration, but regret magnitudes
    /// **grow** with `t` instead of staying bounded (β=1 discounts negatives the
    /// same way as positives), which is the regime where low-precision integer
    /// regret storage works ([`LeanTable`](crate::solver::lean_table::LeanTable)
    /// — and the reason Pluribus stored its blueprint as ints).  The bounded
    /// regrets of `RECOMMENDED` + quantized storage was tried and measurably
    /// broke convergence (see the note atop `regret_table.rs`).
    pub const LINEAR: Discount = Discount { alpha: 1.0, beta: 1.0, gamma: 1.0 };

    /// Multiplicative factor applied to accumulated **positive** regret at the
    /// start of iteration `t`.
    pub fn positive_factor(&self, t: u64) -> f64 {
        let x = (t as f64).powf(self.alpha);
        x / (x + 1.0)
    }

    /// Multiplicative factor applied to accumulated **negative** regret at the
    /// start of iteration `t`.  With `β = 0` this is the constant `0.5`.
    pub fn negative_factor(&self, t: u64) -> f64 {
        let x = (t as f64).powf(self.beta);
        x / (x + 1.0)
    }

    /// Weight applied to this iteration's contribution to the strategy sum.
    pub fn strategy_weight(&self, t: u64) -> f64 {
        (t as f64).powf(self.gamma)
    }
}

/// Parsed from `"α,β,γ"`, e.g. `"1.5,0.5,2"` (the trainer's `--discount`).
impl std::str::FromStr for Discount {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        let parts: Vec<f64> = s
            .split(',')
            .map(|p| p.trim().parse::<f64>().map_err(|_| format!("bad number {p:?} in {s:?}")))
            .collect::<Result<_, _>>()?;
        match parts[..] {
            [alpha, beta, gamma] if parts.iter().all(|x| x.is_finite() && *x >= 0.0) => {
                Ok(Discount { alpha, beta, gamma })
            }
            _ => Err(format!("discount {s:?}: give three non-negative numbers α,β,γ")),
        }
    }
}

impl std::fmt::Display for Discount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{},{},{}", self.alpha, self.beta, self.gamma)
    }
}

#[cfg(test)]
mod parse_tests {
    use super::Discount;

    #[test]
    fn discount_parses_three_numbers_and_prints_back() {
        let d: Discount = "1.5,0.5,2".parse().unwrap();
        assert_eq!((d.alpha, d.beta, d.gamma), (1.5, 0.5, 2.0));
        assert_eq!(d.to_string(), "1.5,0.5,2");
        assert_eq!(Discount::RECOMMENDED.to_string(), "1.5,0,2");
        for bad in ["1.5,0.5", "1.5,x,2", "1.5,-1,2", "1,2,3,4"] {
            assert!(bad.parse::<Discount>().is_err(), "{bad} should not parse");
        }
    }
}
