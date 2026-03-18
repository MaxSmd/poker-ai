//! Info set → bucket lookup.
//!
//! Loaded at solver startup from `data/abstraction.bin`.

pub struct BucketMap {
    // TODO: internal map from (street, hole_cards, board) → bucket id
}

impl BucketMap {
    /// Load the abstraction map from a bincode file.
    pub fn load(_path: &str) -> std::io::Result<Self> {
        todo!()
    }

    /// Look up the bucket id for a given info set.
    pub fn bucket(&self, _street: u8, _hole_cards: &[u8; 2], _board: &[u8]) -> u32 {
        todo!()
    }
}
