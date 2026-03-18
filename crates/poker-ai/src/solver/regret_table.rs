//! Structure-of-Arrays regret storage.
//!
//! Layout: flat Vec<f32> for regrets and strategy_sum, indexed via offsets.

pub struct RegretTable {
    /// Flat regret array: [infoset_0_action_0, infoset_0_action_1, ...]
    pub regrets: Vec<f32>,
    /// Same layout as regrets.
    pub strategy_sum: Vec<f32>,
    /// Number of actions per info set.
    pub num_actions: Vec<u8>,
    /// Start index in regrets/strategy_sum for each info set.
    pub offsets: Vec<u32>,
}

impl RegretTable {
    pub fn new() -> Self {
        Self {
            regrets: Vec::new(),
            strategy_sum: Vec::new(),
            num_actions: Vec::new(),
            offsets: Vec::new(),
        }
    }

    /// Get the regret slice for a given info set.
    pub fn regrets_for(&self, info_set: usize) -> &[f32] {
        let start = self.offsets[info_set] as usize;
        let len = self.num_actions[info_set] as usize;
        &self.regrets[start..start + len]
    }

    /// Get the mutable regret slice for a given info set.
    pub fn regrets_for_mut(&mut self, info_set: usize) -> &mut [f32] {
        let start = self.offsets[info_set] as usize;
        let len = self.num_actions[info_set] as usize;
        &mut self.regrets[start..start + len]
    }
}
