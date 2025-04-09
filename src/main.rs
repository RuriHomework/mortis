pub mod board;
pub mod piece;
use board::{BOARD_HEIGHT, BOARD_WIDTH, Board, FEATURES, WEIGHTS};
use cmaes::{CMAESOptions, DVector, Mode, PlotOptions};
use piece::{PieceType, ROTATIONS};
use rand::Rng;
use std::env;
use std::{thread, time::Duration};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() <= 1 {
        println!("Usage: tetris [preview|train <generations>]");
        println!("  preview: Show AI gameplay visualization");
        println!("  train: Train the AI with specified generations");
        return;
    }

    match args[1].as_str() {
        "preview" => preview(),
        "train" => {
            let generations = if args.len() > 2 {
                args[2].parse().unwrap_or(20)
            } else {
                20
            };
            let target = if args.len() > 3 {
                args[3].parse().unwrap_or(1_000_000.0)
            } else {
                1_000_000.0
            };
            train(generations, target);
        }
        _ => {
            println!("Unknown command. Use 'preview' or 'train'");
        }
    }
}

fn train(generations: usize, target: f64) {
    println!("开始使用CMAES训练俄罗斯方块AI参数...");

    let running = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        println!("\n接收到Ctrl+C, 正在结束训练...");
        r.store(false, std::sync::atomic::Ordering::SeqCst);
    })
    .expect("Error setting Ctrl+C handler");

    let objective_function = |weights: &DVector<f64>| {
        let mut weights_array = [0.0; FEATURES];
        for i in 0..FEATURES {
            weights_array[i] = weights[i];
        }

        let norm = weights_array.iter().map(|w| w.powi(2)).sum::<f64>().sqrt();
        if norm > 0.0 {
            for w in &mut weights_array {
                *w /= norm;
            }
        }

        let num_games = 20;
        let mut total_score = 0.0;

        for _ in 0..num_games {
            let score = simulate_game(&weights_array);
            total_score += score as f64;
        }

        let avg_score = total_score / num_games as f64;

        avg_score
    };

    let initial_weights = DVector::from_vec(vec![0.0; FEATURES]);
    // let initial_weights = WEIGHTS.to_vec();
    let initial_step_size = 0.5;

    let mut cmaes_states = CMAESOptions::new(initial_weights, initial_step_size)
        .mode(Mode::Maximize)
        .max_generations(generations)
        .weights(cmaes::Weights::Negative)
        .population_size(240)
        .enable_plot(PlotOptions::new(0, false))
        .enable_printing(50)
        .build(objective_function)
        .unwrap();

    println!("正在运行CMAES优化, 总共{}代...", generations);

    'main: loop {
        let result = loop {
            if let Some(data) = cmaes_states.next_parallel() {
                break data;
            }

            if !running.load(std::sync::atomic::Ordering::SeqCst) {
                cmaes_states
                    .get_plot()
                    .unwrap()
                    .save_to_file("plot.png", true)
                    .unwrap();
                println!("优化完成！");
                print_results(&cmaes_states.current_best_individual().unwrap());
                break 'main;
            }
        };

        if !running.load(std::sync::atomic::Ordering::SeqCst) {
            cmaes_states
                .get_plot()
                .unwrap()
                .save_to_file("plot.png", true)
                .unwrap();
            println!("优化完成！");
            print_results(&cmaes_states.current_best_individual().unwrap());
            break 'main;
        }

        if cmaes_states.generation() > generations {
            cmaes_states
                .get_plot()
                .unwrap()
                .save_to_file("plot.png", true)
                .unwrap();

            println!("优化完成！");
            print_results(&result.current_best.unwrap());
            break 'main;
        }

        if let Some(ref best) = result.overall_best {
            if best.value > target {
                cmaes_states
                    .get_plot()
                    .unwrap()
                    .save_to_file("plot.png", true)
                    .unwrap();

                println!("优化完成！");
                print_results(&result.current_best.unwrap());
                break 'main;
            }
        };
    }
}

fn print_results(best: &cmaes::Individual) {
    println!("最佳分数: {:.2}", best.value);

    println!("最佳权重数组形式:");
    print!("[");
    for (i, &w) in best.point.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{:.6}", w);
    }
    println!("]");
}

fn simulate_game(weights: &[f64; FEATURES]) -> i32 {
    let mut board = Board::new();
    let mut rng = rand::rng();

    let num_pieces = 1_000_000;

    for _ in 0..num_pieces {
        let piece_type = match rng.random_range(0..7) {
            0 => PieceType::I,
            1 => PieceType::T,
            2 => PieceType::O,
            3 => PieceType::J,
            4 => PieceType::L,
            5 => PieceType::S,
            _ => PieceType::Z,
        };

        let mut possible_actions = Vec::new();
        for rotate in 0..4 {
            let p = &ROTATIONS[piece_type as usize][rotate];
            for x in 0..=(BOARD_WIDTH - p.width) {
                if let Some((_, features)) = board.simulate(piece_type, x, rotate) {
                    let action_score = features
                        .iter()
                        .zip(weights.iter())
                        .map(|(f, w)| f * w)
                        .sum::<f64>();
                    possible_actions.push((rotate, x, action_score));
                }
            }
        }

        if possible_actions.is_empty() {
            break;
        }

        possible_actions.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        let best_action = possible_actions[0];

        board
            .apply(piece_type, best_action.1, best_action.0)
            .unwrap();
    }

    board.get_score()
}

fn preview() {
    let mut board = Board::new();
    let mut rng = rand::rng();
    let piece_symbols = ['I', 'T', 'O', 'J', 'L', 'S', 'Z'];
    let piece_colors = [
        "\x1B[36m", "\x1B[35m", "\x1B[33m", "\x1B[34m", "\x1B[31m", "\x1B[32m", "\x1B[91m",
    ];

    println!("Tetris AI Preview (按Ctrl+C退出)");

    let mut current_piece_type = get_random_piece(&mut rng);
    let mut next_piece_type = get_random_piece(&mut rng);

    loop {
        let mut possible_actions = Vec::new();
        for rotate in 0..4 {
            let p = &ROTATIONS[current_piece_type as usize][rotate];
            for x in 0..=(BOARD_WIDTH - p.width) {
                if let Some((_, features)) = board.simulate(current_piece_type, x, rotate) {
                    let action_score = features
                        .iter()
                        .zip(WEIGHTS.iter())
                        .map(|(f, w)| f * w)
                        .sum::<f64>();
                    possible_actions.push((rotate, x, action_score));
                }
            }
        }

        if possible_actions.is_empty() {
            println!("游戏结束！无法放置方块: {:?}", current_piece_type);
            break;
        }

        possible_actions.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        let best_action = possible_actions[0];

        board
            .apply(current_piece_type, best_action.1, best_action.0)
            .unwrap();

        print!("\x1B[2J\x1B[1;1H");

        println!("╔══════════════════════════════════════╗");
        println!("║ Tetris AI Preview - Score: {:<9} ║", board.get_score());
        println!("╚══════════════════════════════════════╝");

        display_game_with_next_piece(
            &board,
            current_piece_type,
            next_piece_type,
            best_action,
            piece_symbols,
            piece_colors,
        );

        current_piece_type = next_piece_type;
        next_piece_type = get_random_piece(&mut rng);

        thread::sleep(Duration::from_millis(10));
    }
}

fn get_random_piece(rng: &mut impl Rng) -> PieceType {
    match rng.random_range(0..7) {
        0 => PieceType::I,
        1 => PieceType::T,
        2 => PieceType::O,
        3 => PieceType::J,
        4 => PieceType::L,
        5 => PieceType::S,
        _ => PieceType::Z,
    }
}

fn display_game_with_next_piece(
    board: &Board,
    current_piece: PieceType,
    next_piece: PieceType,
    best_action: (usize, usize, f64),
    piece_symbols: [char; 7],
    piece_colors: [&str; 7],
) {
    let grid = board.get_grid();
    let color_grid = board.get_color_grid();

    let next_piece_shape = &ROTATIONS[next_piece as usize][0];
    let next_piece_color = piece_colors[next_piece as usize];

    let mut next_preview = [[false; 4]; 4];

    let offset_x = (4 - next_piece_shape.width) / 2;
    let offset_y = 1;

    for y in 0..next_piece_shape.height {
        for x in 0..next_piece_shape.width {
            if y + offset_y < 4 && x + offset_x < 4 && next_piece_shape.shape[y][x] != 0 {
                next_preview[y + offset_y][x + offset_x] = true;
            }
        }
    }

    println!("╔{}╗    ╔══════╗", "═".repeat(BOARD_WIDTH));
    println!("║{}║    ║ NEXT ║", " ".repeat(BOARD_WIDTH));
    println!("║{}║    ╠══════╣", " ".repeat(BOARD_WIDTH));
    println!("║{}║    ║      ║", " ".repeat(BOARD_WIDTH));
    println!("║{}║    ║      ║", " ".repeat(BOARD_WIDTH));
    println!("║{}║    ║      ║", " ".repeat(BOARD_WIDTH));

    for y in (0..BOARD_HEIGHT).rev() {
        print!("║");

        for x in 0..BOARD_WIDTH {
            if grid[y][x] {
                let color_index = color_grid[y][x].unwrap_or(0) as usize;
                let color_code = if color_index < piece_colors.len() {
                    piece_colors[color_index]
                } else {
                    "\x1B[37m"
                };
                print!("{}\u{25A0}\x1B[0m", color_code);
            } else {
                print!(" ");
            }
        }

        let preview_row = BOARD_HEIGHT - y - 1;
        if preview_row < 6 {
            print!("║    ║ ");

            if preview_row >= 1 && preview_row <= 4 {
                let row_idx = preview_row - 1;
                for col in 0..4 {
                    if next_preview[row_idx][col] {
                        print!("{}\u{25A0}\x1B[0m", next_piece_color);
                    } else {
                        print!(" ");
                    }
                }
            } else {
                print!("    ");
            }

            print!(" ║");
        } else {
            print!("║    ║      ║");
        }

        println!();
    }

    println!("╚{}╝    ╚══════╝", "═".repeat(BOARD_WIDTH));

    println!(
        "当前: {}{}{}(旋转: {}, 位置: {})",
        piece_colors[current_piece as usize],
        piece_symbols[current_piece as usize],
        "\x1B[0m",
        best_action.0,
        best_action.1
    );
}
