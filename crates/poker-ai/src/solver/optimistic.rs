//! Optimistic regret update (momentum term).
//!
//! Farina et al., ICML 2021.
//! R_t = R_{t-1} + 2*r_t - r_{t-1}

/// Apply the optimistic update.
/// `regrets` — current cumulative regrets (mutated in place)
/// `prev_instantaneous` — instantaneous regrets from the previous iteration
/// `curr_instantaneous` — instantaneous regrets from the current iteration
pub fn optimistic_update(
    regrets: &mut [f32],
    prev_instantaneous: &[f32],
    curr_instantaneous: &[f32],
) {
    for ((r, &prev), &curr) in regrets
        .iter_mut()
        .zip(prev_instantaneous.iter())
        .zip(curr_instantaneous.iter())
    {
        *r += 2.0 * curr - prev;
    }
}
