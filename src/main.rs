use streams_solver::{ GameState, McParams, ev_before_draw, SimpleRng, board_from_str };
/*────────────── CLI テスト ─────────────*/
#[cfg(not(target_arch = "wasm32"))]
fn main() {

    let arg: Vec<String> = std::env::args().collect();
    if arg.len() != 2 {
        eprintln!("usage: {} <board20>", arg[0]);
        std::process::exit(1);
    }
    let board = board_from_str(&arg[1]).expect("board");
    let mut st = GameState::new(board);
    let p = McParams::default();
    let mut rng = SimpleRng::default();
    println!("EV = {:.3}", ev_before_draw(&mut st, &p, &mut rng, 0));
}