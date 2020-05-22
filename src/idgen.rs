pub struct IDGenerator {
    id: u32,
}

impl IDGenerator {
    pub fn new() -> Self {
        IDGenerator {
            id: 0
        }
    }

    pub fn get(&mut self) -> u32 {
        self.id = self.id + 1;
        self.id - 1
    }
}