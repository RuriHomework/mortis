use anyhow::Result;

use crate::piece::{PieceType, ROTATIONS};
pub const BOARD_HEIGHT: usize = 15;
pub const BOARD_WIDTH: usize = 10;
pub const FEATURES: usize = 13;

pub static WEIGHTS: [f64; FEATURES] = [
    148226.044742,
    -235469.293532,
    227818.466659,
    28075.637356,
    -151691.585200,
    168940.778157,
    -924.939634,
    902941.598550,
    -81899.099075,
    229232.505421,
    196865.056503,
    19932.300712,
    185679.872248,
];

pub struct Board {
    pub grid: [[bool; BOARD_WIDTH]; BOARD_HEIGHT],
    pub color_grid: [[Option<u8>; BOARD_WIDTH]; BOARD_HEIGHT],
    pub heights: [usize; BOARD_WIDTH],
    pub score: i32,
}

impl Board {
    pub fn new() -> Self {
        Board {
            grid: [[false; BOARD_WIDTH]; BOARD_HEIGHT],
            color_grid: [[None; BOARD_WIDTH]; BOARD_HEIGHT],
            heights: [0; BOARD_WIDTH],
            score: 0,
        }
    }

    pub fn get_height(&self, col: usize) -> usize {
        self.heights[col]
    }

    pub fn get_grid(&self) -> &[[bool; BOARD_WIDTH]; BOARD_HEIGHT] {
        &self.grid
    }

    pub fn get_color_grid(&self) -> &[[Option<u8>; BOARD_WIDTH]; BOARD_HEIGHT] {
        &self.color_grid
    }

    pub fn get_score(&self) -> i32 {
        self.score
    }

    pub fn simulate(
        &self,
        piece_type: PieceType,
        x: usize,
        rotate: usize,
    ) -> Option<(i32, [f64; FEATURES])> {
        let piece = &ROTATIONS[piece_type as usize][rotate];

        // Check x boundaries
        if x + piece.width > BOARD_WIDTH {
            return None;
        }

        // Calculate required y position
        let mut required_y = 0;
        for dx in 0..piece.width {
            let col = x + dx;
            let h_col = self.heights[col];
            let mut max_i_for_dx = 0;
            let mut has_block = false;

            for i in 0..piece.height {
                if piece.shape[i][dx] != 0 {
                    has_block = true;
                    let current_required_y = h_col as i32 - i as i32;
                    if current_required_y > max_i_for_dx {
                        max_i_for_dx = current_required_y;
                    }
                }
            }

            if has_block && max_i_for_dx > required_y {
                required_y = max_i_for_dx;
            }
        }
        let required_y = required_y as usize;

        // Check if piece fits
        let mut blocks = Vec::new();
        for i in 0..piece.height {
            for j in 0..piece.width {
                if piece.shape[i][j] != 0 {
                    let y = required_y + i;
                    let col = x + j;
                    if y >= BOARD_HEIGHT || self.grid[y][col] {
                        return None;
                    }
                    blocks.push((y, col));
                }
            }
        }

        // Create temporary grid and heights
        let mut temp_grid = self.grid.clone();
        let mut temp_heights = self.heights.clone();

        // Place the piece
        let mut max_h = 0;
        for &(y, col) in &blocks {
            temp_grid[y][col] = true;
            temp_heights[col] = temp_heights[col].max(y + 1);
            max_h = max_h.max(y + 1);
        }

        // Check for full rows
        let mut full_rows = Vec::new();
        for y in 0..BOARD_HEIGHT {
            if (0..BOARD_WIDTH).all(|x| temp_grid[y][x]) {
                full_rows.push(y);
            }
        }
        let cleared = full_rows.len() as i32;

        // Clear full rows if any
        if !full_rows.is_empty() {
            let mut new_grid = [[false; BOARD_WIDTH]; BOARD_HEIGHT];
            let mut shift = 0;

            for y in (0..BOARD_HEIGHT).rev() {
                if shift < full_rows.len() && y == full_rows[full_rows.len() - 1 - shift] {
                    shift += 1;
                    continue;
                }

                let new_y = y + shift;
                if new_y < BOARD_HEIGHT {
                    new_grid[new_y] = temp_grid[y];
                }
            }

            temp_grid = new_grid;

            // Recalculate heights
            temp_heights = [0; BOARD_WIDTH];
            for x in 0..BOARD_WIDTH {
                for y in (0..BOARD_HEIGHT).rev() {
                    if temp_grid[y][x] {
                        temp_heights[x] = y + 1;
                        break;
                    }
                }
            }
        }

        // Calculate features
        let mut features = [0.0; FEATURES];

        // 1. landing_height (highest block's y coordinate)
        let landing_height = blocks.iter().map(|&(y, _)| y).max().unwrap_or(0);
        features[0] = landing_height as f64;

        // 2. eroded_piece_cells (number of blocks in cleared rows × cleared rows)
        let mut eroded = 0;
        for &(y, _) in &blocks {
            if full_rows.contains(&y) {
                eroded += 1;
            }
        }
        features[1] = (eroded * cleared) as f64;

        // 3. row_transitions (row transitions)
        let mut row_trans = 0;
        for y in 0..BOARD_HEIGHT {
            let mut prev = true;
            let mut cnt = 0;
            for x in 0..BOARD_WIDTH {
                let curr = temp_grid[y][x];
                if curr != prev {
                    cnt += 1;
                }
                prev = curr;
            }
            if !prev {
                cnt += 1;
            }
            row_trans += cnt;
        }
        features[2] = row_trans as f64;

        // 4. column_transitions (column transitions)
        let mut col_trans = 0;
        for x in 0..BOARD_WIDTH {
            let mut prev = true;
            let mut cnt = 0;
            for y in 0..BOARD_HEIGHT {
                let curr = temp_grid[y][x];
                if curr != prev {
                    cnt += 1;
                }
                prev = curr;
            }
            if !prev {
                cnt += 1;
            }
            col_trans += cnt;
        }
        features[3] = col_trans as f64;

        // 5. holes (number of holes)
        let mut holes = 0;
        for x in 0..BOARD_WIDTH {
            let mut top = None;
            for y in (0..BOARD_HEIGHT).rev() {
                if temp_grid[y][x] {
                    top = Some(y);
                    break;
                }
            }
            if let Some(top_y) = top {
                for y in 0..top_y {
                    if !temp_grid[y][x] {
                        holes += 1;
                    }
                }
            }
        }
        features[4] = holes as f64;

        // 6. board_wells (well sums)
        let mut wells = 0;
        for x in 0..BOARD_WIDTH {
            let left = if x > 0 {
                temp_heights[x - 1]
            } else {
                temp_heights[x]
            };
            let right = if x < BOARD_WIDTH - 1 {
                temp_heights[x + 1]
            } else {
                temp_heights[x]
            };
            let current = temp_heights[x];
            if current < left && current < right {
                wells += left.min(right) - current;
            }
        }
        features[5] = wells as f64;

        // 7. hole_depth (hole depth)
        let mut hole_depth = 0;
        for x in 0..BOARD_WIDTH {
            let current_h = temp_heights[x];
            for y in 0..current_h {
                if !temp_grid[y][x] {
                    hole_depth += current_h - y;
                }
            }
        }
        features[6] = hole_depth as f64;

        // 8. rows_with_holes (rows with holes)
        let mut rows_with_holes = 0;
        for y in 0..BOARD_HEIGHT {
            let mut has_hole = false;
            for x in 0..BOARD_WIDTH {
                if !temp_grid[y][x] && temp_heights[x] > y {
                    has_hole = true;
                    break;
                }
            }
            if has_hole {
                rows_with_holes += 1;
            }
        }
        features[7] = rows_with_holes as f64;

        // 9. diversity
        let mut diversity = 0;
        let mut prev_h = temp_heights[0];
        for x in 1..BOARD_WIDTH {
            diversity += ((temp_heights[x] - prev_h) as i32).abs();
            prev_h = temp_heights[x];
        }
        features[8] = diversity as f64;

        // 10. RFB
        let c =
            (0..BOARD_WIDTH).map(|i| temp_heights[i]).sum::<usize>() as f64 / BOARD_WIDTH as f64;
        let h = BOARD_HEIGHT as f64;
        for i in 0..4 {
            let term = c - (i as f64 * h / 3.0);
            features[9 + i] = (-term.powi(2) / (2.0 * (h / 5.0).powi(2))).exp();
        }

        Some((cleared, features))
    }

    pub fn apply(
        &mut self,
        piece_type: PieceType,
        x: usize,
        rotate: usize,
    ) -> Result<(), &'static str> {
        let piece = &ROTATIONS[piece_type as usize][rotate];
        let color = piece_type as u8;

        // Check x boundaries
        if x + piece.width > BOARD_WIDTH {
            return Err("Piece out of bounds");
        }

        // Calculate required y position
        let mut required_y = 0;
        for dx in 0..piece.width {
            let col = x + dx;
            let h_col = self.heights[col];
            let mut max_i_for_dx = 0;
            let mut has_block = false;

            for i in 0..piece.height {
                if piece.shape[i][dx] != 0 {
                    has_block = true;
                    let current_required_y = h_col as i32 - i as i32;
                    if current_required_y > max_i_for_dx {
                        max_i_for_dx = current_required_y;
                    }
                }
            }

            if has_block && max_i_for_dx > required_y {
                required_y = max_i_for_dx;
            }
        }
        let required_y = required_y as usize;

        // Check if piece fits and collect blocks
        let mut blocks = Vec::new();
        for i in 0..piece.height {
            for j in 0..piece.width {
                if piece.shape[i][j] != 0 {
                    let y = required_y + i;
                    let col = x + j;
                    if y >= BOARD_HEIGHT || self.grid[y][col] {
                        return Err("Piece doesn't fit");
                    }
                    blocks.push((y, col));
                }
            }
        }

        // Place the piece
        let mut max_h = 0;
        for &(y, col) in &blocks {
            self.grid[y][col] = true;
            self.color_grid[y][col] = Some(color);
            self.heights[col] = self.heights[col].max(y + 1);
            max_h = max_h.max(y + 1);
        }

        // Check for full rows
        let mut full_rows = Vec::new();
        for y in 0..BOARD_HEIGHT {
            if (0..BOARD_WIDTH).all(|x| self.grid[y][x]) {
                full_rows.push(y);
            }
        }

        // Clear full rows if any
        if !full_rows.is_empty() {
            let mut new_grid = [[false; BOARD_WIDTH]; BOARD_HEIGHT];
            let mut new_color_grid = [[None; BOARD_WIDTH]; BOARD_HEIGHT];
            let mut shift = 0;

            for y in 0..BOARD_HEIGHT {
                if full_rows.contains(&y) {
                    shift += 1;
                    continue;
                }

                let new_y = y - shift;
                if new_y < BOARD_HEIGHT {
                    new_grid[new_y] = self.grid[y];
                    new_color_grid[new_y] = self.color_grid[y];
                }
            }

            self.grid = new_grid;
            self.color_grid = new_color_grid;

            // Recalculate heights
            self.heights = [0; BOARD_WIDTH];
            for x in 0..BOARD_WIDTH {
                for y in (0..BOARD_HEIGHT).rev() {
                    if self.grid[y][x] {
                        self.heights[x] = y + 1;
                        break;
                    }
                }
            }

            // Update score
            let add_score = match full_rows.len() {
                1 => 100,
                2 => 300,
                3 => 500,
                4 => 800,
                _ => 0,
            };
            self.score += add_score;
        }

        Ok(())
    }

    pub fn get_start_y(&mut self, piece_type: PieceType, x: usize, rotate: usize) -> usize {
        let piece = &ROTATIONS[piece_type as usize][rotate];

        // Check x boundaries using leftmost and rightmost
        let left = x as i32 + piece.leftmost[rotate];
        let right = x as i32 + piece.rightmost[rotate];
        if left < 0 || right >= BOARD_WIDTH as i32 {
            return 0;
        }

        // Find maximum i where shape has non-zero elements
        let i_max = (0..4)
            .rev()
            .find(|&i| piece.shape[i].iter().any(|&v| v != 0))
            .unwrap_or(0);

        // Calculate starting y based on minimum column height
        let columns = x..x + piece.width;
        let min_height = columns.clone().map(|j| self.heights[j]).min().unwrap();
        min_height - i_max
    }

    #[allow(dead_code)]
    pub fn draw(&self) {
        print!("\x1B[2J\x1B[1;1H");
        println!("Score: {}", self.score);
        for row in self.grid.iter().rev() {
            print!("|");
            for &cell in row {
                print!("{}", if cell { "■" } else { " " });
            }
            println!("|");
        }
        println!("--------------------------");
    }

    pub fn draw_colored(&self) {
        print!("\x1B[2J\x1B[1;1H");
        println!("Score: {}", self.score);

        // Print top border
        println!("╔{}╗", "═".repeat(BOARD_WIDTH));

        // Print each row
        for y in (0..BOARD_HEIGHT).rev() {
            print!("║");
            for x in 0..BOARD_WIDTH {
                if self.grid[y][x] {
                    let color_code = match self.color_grid[y][x] {
                        Some(0) => "\x1B[36m", // Cyan - I type
                        Some(1) => "\x1B[35m", // Purple - T type
                        Some(2) => "\x1B[33m", // Yellow - O type
                        Some(3) => "\x1B[34m", // Blue - J type
                        Some(4) => "\x1B[31m", // Red - L type
                        Some(5) => "\x1B[32m", // Green - S type
                        Some(6) => "\x1B[91m", // Bright Red - Z type
                        _ => "\x1B[37m",       // White - Default
                    };
                    print!("{}\u{25A0}\x1B[0m", color_code);
                } else {
                    print!(" ");
                }
            }
            println!("║");
        }

        // Print bottom border
        println!("╚{}╝", "═".repeat(BOARD_WIDTH));
    }
}
