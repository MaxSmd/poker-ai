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

use crate::util::rng::Rng;

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
    let mut rng = Rng::new(seed);

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

/// Globally-optimal 1-D clustering into `k` contiguous groups (Ckmeans dynamic
/// program), for a **scalar** feature such as river equity.
///
/// K-means on a scalar is the wrong tool — Lloyd's only finds a local optimum and
/// (on the river's near-degenerate distribution) collapses into incoherent
/// catch-all buckets.  In one dimension the optimum is a set of contiguous
/// intervals, found exactly by the O(k·m²) DP that minimises the weighted
/// within-cluster sum of squares.  `values` must be sorted ascending and
/// distinct; `weights[i]` is how many situations carry `values[i]`.  Returns the
/// cluster id (`0..k`, non-decreasing with value) for each input value.
/// Deterministic — no seed, no iteration count.
pub fn cluster_1d(values: &[f64], weights: &[u64], k: usize) -> Vec<usize> {
    let m = values.len();
    assert_eq!(weights.len(), m, "values and weights must have equal length");
    assert!(m >= 1 && k >= 1, "need ≥1 value and ≥1 cluster");
    debug_assert!(values.windows(2).all(|w| w[0] < w[1]), "values must be sorted and distinct");

    let k = k.min(m);
    if k == 1 {
        return vec![0; m];
    }
    if k == m {
        return (0..m).collect();
    }

    // Prefix sums of w, w·x, w·x² → O(1) weighted SSE of any contiguous range.
    let (mut pw, mut pwx, mut pwx2) = (vec![0f64; m + 1], vec![0f64; m + 1], vec![0f64; m + 1]);
    for i in 0..m {
        let (w, x) = (weights[i] as f64, values[i]);
        pw[i + 1] = pw[i] + w;
        pwx[i + 1] = pwx[i] + w * x;
        pwx2[i + 1] = pwx2[i] + w * x * x;
    }
    // Weighted SSE of points in [a, b): Σw·x² − (Σw·x)²/Σw.
    let cost = |a: usize, b: usize| -> f64 {
        let w = pw[b] - pw[a];
        if w <= 0.0 {
            return 0.0;
        }
        let sx = pwx[b] - pwx[a];
        (pwx2[b] - pwx2[a]) - sx * sx / w
    };

    // d[j][i] = min cost of clustering the first i values into j clusters;
    // split[j][i] = start of the last (j-th) cluster.
    let mut d = vec![vec![f64::INFINITY; m + 1]; k + 1];
    let mut split = vec![vec![0usize; m + 1]; k + 1];
    d[0][0] = 0.0;
    for j in 1..=k {
        for i in j..=m {
            for t in (j - 1)..i {
                let c = d[j - 1][t] + cost(t, i);
                if c < d[j][i] {
                    d[j][i] = c;
                    split[j][i] = t;
                }
            }
        }
    }

    let mut assign = vec![0usize; m];
    let mut i = m;
    for j in (1..=k).rev() {
        let t = split[j][i];
        for a in assign.iter_mut().take(i).skip(t) {
            *a = j - 1;
        }
        i = t;
    }
    assign
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cluster_1d_is_contiguous_monotone_and_deterministic() {
        let values = [0.0, 0.1, 0.12, 0.5, 0.52, 0.9, 0.95];
        let weights = [1u64; 7];
        let a = cluster_1d(&values, &weights, 3);
        let b = cluster_1d(&values, &weights, 3);
        assert_eq!(a, b, "deterministic");
        // Non-decreasing with value, and exactly 3 contiguous groups.
        assert!(a.windows(2).all(|w| w[0] <= w[1]), "monotone");
        assert_eq!(*a.iter().max().unwrap(), 2, "uses all 3 clusters");
        // The three natural groups (≈0, ≈0.5, ≈0.9) must each be one cluster.
        assert_eq!(a[0], a[2]);
        assert_eq!(a[3], a[4]);
        assert_eq!(a[5], a[6]);
        assert_ne!(a[0], a[3]);
        assert_ne!(a[3], a[5]);
    }

    #[test]
    fn cluster_1d_trivial_cases() {
        let v = [0.1, 0.2, 0.3, 0.4];
        let w = [1u64; 4];
        assert_eq!(cluster_1d(&v, &w, 1), vec![0, 0, 0, 0]);
        assert_eq!(cluster_1d(&v, &w, 4), vec![0, 1, 2, 3]);
        // k larger than the number of distinct values clamps to one-per-value.
        assert_eq!(cluster_1d(&v, &w, 9), vec![0, 1, 2, 3]);
    }

    #[test]
    fn cluster_1d_respects_weights() {
        // A heavy mass at 0.30/0.31 should not be split before the lone outliers
        // are separated: with k=2 the optimal split isolates the far point.
        let values = [0.30, 0.31, 0.32, 0.95];
        let weights = [100u64, 100, 100, 1];
        let a = cluster_1d(&values, &weights, 2);
        assert_eq!(a, vec![0, 0, 0, 1], "the dense mass stays together, outlier splits off");
    }

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
