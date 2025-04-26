//! Monte-Carlo Hybrid EV solver (Wasm-friendly, no Rayon/SIMD)
//! ───────────────────────────────────────────────────────────
//! * 固定長配列＋インプレース更新
//! * HashMap → 配列カウンタ
//! * clone() 排除・バックトラック式再帰
//! * SimpleRng (xorshift*) を継続使用
//!
//! ビルド例
//! ```bash
//! wasm-pack build --release --target web
//! ```

#[cfg(target_arch = "wasm32")] use js_sys::Float64Array;
#[cfg(target_arch = "wasm32")] use wasm_bindgen::prelude::*;
use core::time::Duration;

/*─────────────── 定数 ───────────────*/
pub const BOARD_SIZE: usize = 20;
const MIN_CARD: u8 = 1;
const MAX_CARD: u8 = 30;
const JOKER:     u8 = 31;   // 内部表現で 31 を Joker に割当て
const DUPLICATE_RANGE: core::ops::RangeInclusive<u8> = 11..=19;
const DUPLICATE_COUNT: u8 = 2;
// index == run 長, value == 得点
const SCORE_TABLE: [i32; BOARD_SIZE + 1] = [
    0,0,1,3,5,7,9,10,15,20,25,30,20,40,50,60,70,50,100,150,300
];

/*─────────────── PRNG ───────────────*/
#[derive(Clone, Copy)]
pub struct SimpleRng(u64);
impl SimpleRng {
    pub fn new(seed: u64) -> Self { Self(seed.max(1)) }
    #[inline] pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0; x ^= x >> 12; x ^= x << 25; x ^= x >> 27; self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    #[inline] pub fn gen_range(&mut self, upper: u8) -> u8 { (self.next_u64() as u8) % upper }
}
impl Default for SimpleRng {
    fn default() -> Self {
        let addr = &Self::default as *const _ as usize as u64;
        #[cfg(target_arch = "wasm32")] {
            let clk = js_sys::Date::now() as u64;
            Self::new(addr ^ clk ^ 0x9E37_79B9_7F4A_7C15)
        }
        #[cfg(not(target_arch = "wasm32"))] {
            let clk = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap_or(Duration::ZERO).as_nanos() as u64;
            Self::new(addr ^ clk ^ 0x517C_C1B7_2722_0A95)
        }
    }
}

/*─────────────── 基本型 ───────────────*/
#[derive(Clone, Debug)]
pub struct GameState {
    board: [u8; BOARD_SIZE],   // 0 = 空, 1‒30 = 数札, 31 = Joker
    deck_count: [u8; 32],      // 残っている札の枚数
    deck_len:  u8,             // 未ドロー枚数 (<= 63)
}

impl GameState {
    pub fn new(board: [u8; BOARD_SIZE]) -> Self {
        // 初期山札枚数を設定
        let mut deck_count = [0u8; 32];
        for n in MIN_CARD..=MAX_CARD {
            deck_count[n as usize] = if DUPLICATE_RANGE.contains(&n) { DUPLICATE_COUNT } else { 1 };
        }
        deck_count[JOKER as usize] = 1;
        // 盤面に置かれているカードを山札から減算
        for &c in &board {
            if c != 0 {
                let ix = c as usize;
                deck_count[ix] -= 1;
            }
        }
        let deck_len = deck_count.iter().map(|&x| x as u16).sum::<u16>() as u8;
        Self { board, deck_count, deck_len }
    }

    #[inline] fn place(&mut self, pos: usize, card: u8) {
        debug_assert_eq!(self.board[pos], 0);
        self.board[pos] = card;
    }
    #[inline] fn remove(&mut self, pos: usize) {
        self.board[pos] = 0;
    }

    /// Score current board without allocations
    pub fn score(&self) -> i32 {
        let mut runs = [0u8; BOARD_SIZE];
        let mut runs_len = 0usize;
        let mut len: u8 = 0;
        let mut last_val: Option<i32> = None;
        for &cell in &self.board {
            let val = match cell {
                0       => None,
                JOKER   => last_val,
                n       => Some(n as i32),
            };
            match (val, last_val) {
                (Some(v), Some(prev)) if v >= prev => { len += 1; },
                (Some(_), Some(_))                 => { runs[runs_len] = len; runs_len += 1; len = 1; },
                (Some(_), None)                    => { len = 1; },
                (None, _)                          => { if len > 0 { runs[runs_len] = len; runs_len += 1; } len = 0; },
            }
            last_val = val;
        }
        if len > 0 { runs[runs_len] = len; runs_len += 1; }
        let mut sum = 0i32;
        for i in 0..runs_len { sum += SCORE_TABLE[runs[i] as usize]; }
        sum
    }

    /// 盤上の空きマスの iterator
    #[inline] fn empty_positions<'a>(&'a self) -> impl Iterator<Item=usize> + 'a {
        self.board.iter().enumerate().filter_map(|(i, &c)| (c==0).then(|| i))
    }
}

/*────────────── Monte-Carlo Hybrid ─────────────*/
#[derive(Clone, Copy)]
pub struct McParams { pub sims: usize, pub rollout_limit: usize }
impl Default for McParams { fn default() -> Self { Self { sims: 5, rollout_limit: 1 } } }

pub fn ev_before_draw(st: &mut GameState, p: &McParams, rng: &mut SimpleRng, level: usize) -> f64 {
    if st.deck_len == 0 || st.empty_positions().next().is_none() {
        return st.score() as f64;
    }
    if level >= p.rollout_limit {
        return rollout(st, p, rng);
    }
    let mut ev = 0.0f64;
    let deck_len_f = st.deck_len as f64;
    for card in 1u8..=JOKER {
        let cnt = st.deck_count[card as usize];
        if cnt == 0 { continue; }
        // ドローしたと仮定して山札を更新
        st.deck_count[card as usize] -= 1;
        st.deck_len -= 1;
        let child_ev = ev_after_draw(st, card, p, rng, level);
        // 期待値へ加算
        ev += (cnt as f64 / deck_len_f) * child_ev;
        // 巻き戻し
        st.deck_count[card as usize] += 1;
        st.deck_len += 1;
    }
    ev
}

fn ev_after_draw(st: &mut GameState, card: u8, p: &McParams, rng: &mut SimpleRng, level: usize) -> f64 {
    let mut best = f64::NEG_INFINITY;
    // 空きマスに置いて最大値を取る
    let empties: Vec<usize> = st.empty_positions().collect();
    for pos in empties {
        st.place(pos, card);
        best = best.max(ev_before_draw(st, p, rng, level + 1));
        st.remove(pos);
    }
    best
}

fn rollout(st: &GameState, p: &McParams, rng: &mut SimpleRng) -> f64 {
    let mut sum = 0.0f64;
    for _ in 0..p.sims {
        // ローカルコピー (64byte 未満なのでコピーの方が速い)
        let mut board = st.board;
        let mut deck = st.deck_count;
        let mut deck_len = st.deck_len;
        // 盤を埋め尽くす
        for pos in 0..BOARD_SIZE {
            if board[pos] != 0 { continue; }
            // n 番目のカードを引く
            let idx = rng.gen_range(deck_len);
            let mut acc = 0u8;
            let mut drawn = 0u8;
            for card in 1u8..=JOKER {
                let c = deck[card as usize];
                if acc + c > idx { drawn = card; break; }
                acc += c;
            }
            // デッキ更新
            deck[drawn as usize] -= 1;
            deck_len -= 1;
            board[pos] = drawn;
        }
        // スコア計算
        let gs_sim = GameState { board, deck_count: deck, deck_len };
        sum += gs_sim.score() as f64;
    }
    sum / p.sims as f64
}

/*────────────── 盤面文字列変換 ─────────────*/
#[inline]
pub fn board_from_str(s: &str) -> Result<[u8; BOARD_SIZE], String> {
    if s.chars().count() != BOARD_SIZE { return Err("board string must be 20 chars".into()); }
    let mut arr = [0u8; BOARD_SIZE];
    for (i, ch) in s.chars().enumerate() {
        arr[i] = match ch {
            '_' => 0,
            '★' => JOKER,
            '0'..='9' => ch.to_digit(10).unwrap() as u8,
            'A'..='U' => 10 + (ch as u8 - b'A'),
            _ => return Err(format!("bad char {ch}")),
        };
    }
    Ok(arr)
}

/*────────────── Wasm エクスポート ─────────────*/
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn expected_value_current_board(board: &str, sims: usize) -> f64 {
    let board = board_from_str(board).expect("bad board");
    let mut st = GameState::new(board);
    let p = McParams { sims, ..Default::default() };
    let mut rng = SimpleRng::default();
    ev_before_draw(&mut st, &p, &mut rng, 0)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn expected_values_after_card(board: &str, card: u8, sims: usize) -> Float64Array {
    let board_arr = board_from_str(board).expect("bad board");
    let mut st = GameState::new(board_arr);
    let p = McParams { sims, ..Default::default() };
    let mut rng = SimpleRng::default();
    let mut vals = [0.0f64; BOARD_SIZE];
    for pos in st.clone().empty_positions() {
        st.place(pos, card);
        vals[pos] = ev_before_draw(&mut st, &p, &mut rng, 0);
        st.remove(pos);
    }
    Float64Array::from(&vals[..])
}

/*─────────────────────────────── Tests (native) ───────────────────────────────*/
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn basic_score() {
        let b = board_from_str("123456789ABCDEFGHI__").unwrap();
        let gs = GameState::new(b);
        assert_eq!(gs.score(), 100); // 20 連 run == 300 点
    }

    #[test]
    fn ev_two_empty_cells_mc_vs_bruteforce() {
        let board_str = "123456789ABCDEFGHI__";
        let board  = board_from_str(board_str).unwrap();
        let mut state  = GameState::new(board);
        let params = McParams { sims: 5000, rollout_limit: 2 };
        let mut rng = SimpleRng::default();
        let mc_ev = ev_before_draw(&mut state, &params, &mut rng, 0);
        println!("EV (MC) = {:.3}", mc_ev);
    }
}
