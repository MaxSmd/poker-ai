//! Bucket-quality inspector (Phase 2 deliverable: "inspect bucket contents
//! manually — each bucket should contain hands that feel intuitively similar").
//!
//! Builds a fresh river abstraction on one fixed board — enumerate every hole
//! pair, cluster them, then print each bucket with its equity band and example
//! hands.  A coherent abstraction shows tight per-bucket equity ranges that tile
//! [0, 1] monotonically.
//!
//! Note on the feature: on a *complete* river board the equity distribution is a
//! point mass, so `ehs_histogram` is one-hot and K-means on one-hot vectors
//! degenerates (it produced one incoherent catch-all bucket spanning the whole
//! equity range).  The river is therefore clustered on the **scalar equity**;
//! the multi-bin histogram feature is for the flop and turn, where the runout
//! makes the distribution genuinely spread.
//!
//! Run: cargo run --release --example inspect_buckets [k]

use poker_ai::abstraction::clustering::kmeans;
use poker_ai::abstraction::features::river_equity;
use poker_core::{make_card, rank_of, suit_of};

const RANKS: [char; 13] = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'];
const SUITS: [char; 4] = ['c', 'd', 'h', 's'];

fn card_str(c: u8) -> String {
    format!("{}{}", RANKS[rank_of(c) as usize], SUITS[suit_of(c) as usize])
}

fn hand_str(h: [u8; 2]) -> String {
    let mut h = h;
    h.sort_unstable();
    format!("{}{}", card_str(h[1]), card_str(h[0]))
}

fn main() {
    let k: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(8);

    // River board: A♣ K♦ 9♥ 4♠ 2♣.
    let board =
        [make_card(12, 0), make_card(11, 1), make_card(7, 2), make_card(2, 3), make_card(0, 0)];
    let mut used = 0u64;
    for &c in &board {
        used |= 1 << c;
    }
    let deck: Vec<u8> = (0u8..52).filter(|c| used & (1 << c) == 0).collect();

    // Every hole pair on this board, clustered on its scalar river equity.
    let mut hands: Vec<[u8; 2]> = Vec::new();
    let mut feats: Vec<Vec<f64>> = Vec::new();
    let mut equities: Vec<f64> = Vec::new();
    for i in 0..deck.len() {
        for j in (i + 1)..deck.len() {
            let hole = [deck[i], deck[j]];
            let eq = river_equity(hole, board);
            hands.push(hole);
            feats.push(vec![eq]);
            equities.push(eq);
        }
    }

    let board_str: String = board.iter().map(|&c| card_str(c) + " ").collect();
    println!("Board: {board_str} — {} hands clustered into {k} buckets\n", hands.len());

    let result = kmeans(&feats, k, 1, 100);

    // Gather per-bucket members, then order buckets by mean equity for reading.
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); result.centroids.len()];
    for (idx, &b) in result.assignments.iter().enumerate() {
        buckets[b].push(idx);
    }
    let mut order: Vec<usize> = (0..buckets.len()).collect();
    let mean_eq = |b: &Vec<usize>| -> f64 {
        if b.is_empty() {
            return -1.0;
        }
        b.iter().map(|&i| equities[i]).sum::<f64>() / b.len() as f64
    };
    order.sort_by(|&a, &b| mean_eq(&buckets[a]).partial_cmp(&mean_eq(&buckets[b])).unwrap());

    println!("{:>3}  {:>5}  {:>17}   {}", "blt", "n", "equity min/mean/max", "examples (by equity)");
    for (rank, &b) in order.iter().enumerate() {
        let mut members = buckets[b].clone();
        if members.is_empty() {
            continue;
        }
        members.sort_by(|&x, &y| equities[y].partial_cmp(&equities[x]).unwrap());
        let eqs: Vec<f64> = members.iter().map(|&i| equities[i]).collect();
        let (lo, hi) = (eqs.iter().cloned().fold(1.0f64, f64::min), eqs.iter().cloned().fold(0.0f64, f64::max));
        let mean = mean_eq(&buckets[b]);
        // Show the strongest, a middle, and the weakest hand in the bucket.
        let top = hand_str(hands[members[0]]);
        let mid = hand_str(hands[members[members.len() / 2]]);
        let bot = hand_str(hands[*members.last().unwrap()]);
        println!(
            "{:>3}  {:>5}  {:>5.3} {:>5.3} {:>5.3}   {top}  {mid}  {bot}",
            rank,
            members.len(),
            lo,
            mean,
            hi,
        );
    }
    println!("\n(Each row is one bucket, ordered by mean equity. Tight min..max bands");
    println!(" with monotonically rising means ⇒ a coherent abstraction.)");
}
