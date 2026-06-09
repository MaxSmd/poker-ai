//! K-Means++ clustering with L2 distance on equity histograms.
//!
//! This is the clusterer that turns per-hand equity histograms (from
//! [`super::features`]) into the bucket assignments the solver indexes.  The
//! plan is specific about the choices and the reasons:
//!
//! * **L2 distance on fixed-bin histograms**, not raw Earth Mover's Distance.
//!   EMD has no closed-form centroid (the barycenter needs a linear program per
//!   iteration), whereas L2's centroid is just the mean of the histograms —
//!   trivial and fast — and L2 on a fixed binning is a well-known, empirically
//!   strong proxy for EMD in poker abstraction.
//! * **K-Means++ seeding**, so initial centroids are spread out (chosen with
//!   probability ∝ squared distance to the nearest existing centroid) rather
//!   than clumping and producing dead clusters.
//!
//! The RNG is a seeded `xorshift64*` for reproducible bucketings.
//!
//! **Parallelism.** The two per-point hot loops — the K-Means++ nearest-distance
//! update and Lloyd's assignment step — are run with Rayon across points.  Both
//! are pure argmin/min reductions whose output order is fixed by the point
//! index, so the parallel result is **bit-identical** to the serial one and the
//! bucketing stays reproducible; only the seeding RNG draws stay sequential.

use rayon::prelude::*;

/// Result of a clustering run.
pub struct KMeans {
    /// Cluster index in `0..k` for each input point.
    pub assignments: Vec<usize>,
    /// Final centroid (mean histogram) of each cluster.
    pub centroids: Vec<Vec<f64>>,
}

/// Squared L2 distance between two equal-length vectors.
fn dist_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Cluster `data` (one histogram per row, all the same length) into `k` groups
/// using K-Means++ initialization and Lloyd's algorithm under L2 distance.
///
/// `seed` makes the run reproducible; `max_iters` caps Lloyd's refinement
/// (it also stops early once assignments stop changing).
pub fn kmeans(data: &[Vec<f64>], k: usize, seed: u64, max_iters: usize) -> KMeans {
    assert!(k >= 1, "k must be ≥ 1");
    assert!(!data.is_empty(), "cannot cluster an empty dataset");
    let k = k.min(data.len());
    let dim = data[0].len();
    let mut rng = Rng(seed | 1);

    // ── K-Means++ seeding ────────────────────────────────────────────────────
    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);
    let first = (rng.unit() * data.len() as f64) as usize % data.len();
    centroids.push(data[first].clone());
    // Squared distance from each point to the nearest chosen centroid.
    let mut nearest_sq: Vec<f64> = data.par_iter().map(|p| dist_sq(p, &centroids[0])).collect();
    while centroids.len() < k {
        let total: f64 = nearest_sq.iter().sum();
        let next = if total <= 0.0 {
            // All remaining points coincide with a centroid; pick any.
            (rng.unit() * data.len() as f64) as usize % data.len()
        } else {
            // Sample ∝ D².
            let threshold = rng.unit() * total;
            let mut acc = 0.0;
            let mut chosen = data.len() - 1;
            for (i, &d) in nearest_sq.iter().enumerate() {
                acc += d;
                if acc >= threshold {
                    chosen = i;
                    break;
                }
            }
            chosen
        };
        centroids.push(data[next].clone());
        let last = centroids.last().unwrap();
        nearest_sq
            .par_iter_mut()
            .zip(data.par_iter())
            .for_each(|(ns, p)| *ns = ns.min(dist_sq(p, last)));
    }

    // ── Lloyd's iterations ───────────────────────────────────────────────────
    let mut assignments = vec![0usize; data.len()];
    for _ in 0..max_iters {
        // Assign every point to its nearest centroid (the dominant cost: n×k×dim).
        // Computed in parallel; `collect` preserves point order, so the result is
        // identical to the serial argmin and the bucketing stays reproducible.
        let new_assignments: Vec<usize> = data
            .par_iter()
            .map(|p| {
                let mut best = 0;
                let mut best_d = f64::INFINITY;
                for (c, centroid) in centroids.iter().enumerate() {
                    let d = dist_sq(p, centroid);
                    if d < best_d {
                        best_d = d;
                        best = c;
                    }
                }
                best
            })
            .collect();
        let mut changed = new_assignments != assignments;
        assignments = new_assignments;

        // Recompute centroids as the mean of assigned points.
        let mut sums = vec![vec![0.0; dim]; k];
        let mut counts = vec![0u64; k];
        for (i, p) in data.iter().enumerate() {
            let c = assignments[i];
            counts[c] += 1;
            for (s, &v) in sums[c].iter_mut().zip(p) {
                *s += v;
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for s in &mut sums[c] {
                    *s /= counts[c] as f64;
                }
                centroids[c] = std::mem::take(&mut sums[c]);
            } else {
                // Empty cluster: re-seed it on the point farthest from its own
                // centroid, so no bucket goes to waste.
                let mut worst = 0;
                let mut worst_d = -1.0;
                for (i, p) in data.iter().enumerate() {
                    let d = dist_sq(p, &centroids[assignments[i]]);
                    if d > worst_d {
                        worst_d = d;
                        worst = i;
                    }
                }
                centroids[c] = data[worst].clone();
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    KMeans { assignments, centroids }
}

/// Convenience wrapper matching the original stub signature: cluster with a
/// fixed seed and a reasonable iteration cap, returning just the assignments.
pub fn kmeans_plus_plus(data: &[Vec<f64>], k: usize) -> Vec<usize> {
    kmeans(data, k, 0xC0FF_EE12_3456_789A, 100).assignments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separates_well_separated_clusters() {
        // Three tight blobs far apart in 2-D; k-means must group each blob.
        let blobs = [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]];
        let mut data = Vec::new();
        let mut truth = Vec::new();
        for (b, center) in blobs.iter().enumerate() {
            for j in 0..5 {
                let jitter = j as f64 * 0.01;
                data.push(vec![center[0] + jitter, center[1] - jitter]);
                truth.push(b);
            }
        }
        let result = kmeans(&data, 3, 1, 100);
        // Points sharing a true blob must share a cluster label.
        for i in 0..data.len() {
            for j in 0..data.len() {
                if truth[i] == truth[j] {
                    assert_eq!(result.assignments[i], result.assignments[j], "blob split apart");
                } else {
                    assert_ne!(result.assignments[i], result.assignments[j], "blobs merged");
                }
            }
        }
    }

    #[test]
    fn k_one_puts_everything_in_one_cluster() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let result = kmeans(&data, 1, 7, 50);
        assert!(result.assignments.iter().all(|&c| c == 0));
        // The single centroid is the mean of all points.
        assert!((result.centroids[0][0] - 3.0).abs() < 1e-9);
        assert!((result.centroids[0][1] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn is_deterministic_for_fixed_seed() {
        let data: Vec<Vec<f64>> = (0..20).map(|i| vec![(i % 7) as f64, (i % 3) as f64]).collect();
        let a = kmeans(&data, 4, 99, 100).assignments;
        let b = kmeans(&data, 4, 99, 100).assignments;
        assert_eq!(a, b);
    }

    #[test]
    fn k_capped_at_dataset_size() {
        let data = vec![vec![0.0], vec![1.0]];
        let result = kmeans(&data, 10, 1, 10);
        // Only 2 points, so at most 2 non-empty clusters; assignments are valid.
        assert!(result.assignments.iter().all(|&c| c < 2));
    }

    #[test]
    fn clusters_real_equity_histograms() {
        // End-to-end with the equity layer: a made hand, a draw, and air on the
        // same turn board should land in three different buckets.
        use crate::abstraction::features::ehs_histogram;
        use poker_core::make_card;

        let board = [make_card(11, 0), make_card(7, 0), make_card(2, 0), make_card(9, 1)]; // 3 spades
        let made = [make_card(11, 1), make_card(11, 2)]; // set of kings
        let flush_draw = [make_card(12, 0), make_card(3, 0)]; // nut flush draw (2 spades)
        let air = [make_card(5, 1), make_card(4, 2)]; // nothing

        let data = vec![
            ehs_histogram(&made, &board, 20),
            ehs_histogram(&flush_draw, &board, 20),
            ehs_histogram(&air, &board, 20),
        ];
        let result = kmeans(&data, 3, 3, 100);
        // Three distinct hand types → three distinct buckets.
        let mut labels = result.assignments.clone();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(labels.len(), 3, "three distinct hands should occupy three buckets");
    }
}
