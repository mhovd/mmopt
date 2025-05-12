use faer::Mat;
use mmopt::{ErrorPoly, Predictions};

fn main() {
    // Probabilities
    let probs = vec![0.5, 0.5];

    // Times
    let step = 0.1;
    let start = 0.0;
    let end = 10.0;

    let times: Vec<f64> = (0..=((end / step) as usize))
        .map(|i| start + i as f64 * step)
        .collect();

    let sub1 = times
        .iter()
        .map(|&t| 1.0 * (-1.5 * t).exp())
        .collect::<Vec<f64>>();

    let sub2 = times
        .iter()
        .map(|&t| 1.0 * (-0.25 * t).exp())
        .collect::<Vec<f64>>();

    // Example usage of the structs and methods defined above
    let predictions = Predictions {
        matrix: Mat::from_fn(times.len(), probs.len(), |i, j| {
            if j == 0 {
                sub1[i]
            } else {
                sub2[i]
            }
        }),
        times: times,
        probs: probs,
    };

    let errorpoly = ErrorPoly::new(0.0, 0.05, 0.0, 0.0);

    let res = mmopt::mmopt(&predictions, errorpoly, 1).unwrap();

    println!("Best combo indices: {:?}", res.best_combo_indices);
    println!("Best combo times: {:?}", res.best_combo_times);
    println!("Minimum risk: {:?}", res.min_risk);
}
