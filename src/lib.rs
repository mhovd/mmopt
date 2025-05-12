use faer::Mat;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::fmt::Error;

pub struct Predictions {
    pub matrix: Mat<f64>,
    pub times: Vec<f64>,
    pub probs: Vec<f64>,
}

impl Predictions {
    fn matrix(&self) -> &Mat<f64> {
        &self.matrix
    }

    fn nsub(&self) -> usize {
        self.matrix.ncols()
    }
    fn nout(&self) -> usize {
        self.matrix.nrows()
    }
}

struct CostMatrix {
    matrix: Mat<f64>,
}

impl CostMatrix {
    fn default(predictions: &Predictions) -> Self {
        let cost_matrix = Mat::from_fn(predictions.nsub(), predictions.nsub(), |i, j| {
            if i == j {
                0.0
            } else {
                1.0
            }
        });

        Self {
            matrix: cost_matrix,
        }
    }
}

pub struct ErrorPoly {
    c0: f64,
    c1: f64,
    c2: f64,
    c3: f64,
}

impl ErrorPoly {
    pub fn new(c0: f64, c1: f64, c2: f64, c3: f64) -> Self {
        Self { c0, c1, c2, c3 }
    }

    fn variance(&self, x: f64) -> f64 {
        let sigma = self.c0 + self.c1 * x + self.c2 * x.powi(2) + self.c3 * x.powi(3);
        sigma.powi(2)
    }
}

#[derive(Debug)]
pub struct MmoptResult {
    pub best_combo_indices: Vec<usize>,
    pub best_combo_times: Vec<f64>,
    pub min_risk: f64,
}

pub fn mmopt(
    predictions: &Predictions,
    errorpoly: ErrorPoly,
    nsamp: usize,
) -> Result<MmoptResult, Error> {
    let nsub = predictions.nsub();
    let nout = predictions.nout();

    if nsamp > nout {
        return Err(Error::default()); // Or a more descriptive error
    }
    if nsub == 0 || nout == 0 {
        return Err(Error::default());
    }

    // Create a cost matrix
    let cost_matrix = CostMatrix::default(predictions);

    // Generate sample candidate indices
    let candidate_indices = generate_combinations(nout, nsamp);

    let (best_combo, min_risk) = candidate_indices
        .par_iter()
        .map(|combo| {
            let mut risk = 0.0;
            for i in 0..nsub {
                for j in 0..nsub {
                    if i != j {
                        let i_obs: Vec<f64> = predictions
                            .matrix()
                            .col(i)
                            .iter()
                            .enumerate()
                            .filter_map(|(k, &x)| if combo.contains(&k) { Some(x) } else { None })
                            .collect();

                        let j_obs: Vec<f64> = predictions
                            .matrix()
                            .col(j)
                            .iter()
                            .enumerate()
                            .filter_map(|(k, &x)| if combo.contains(&k) { Some(x) } else { None })
                            .collect();

                        let i_var: Vec<f64> =
                            i_obs.iter().map(|&x| errorpoly.variance(x)).collect();
                        let j_var: Vec<f64> =
                            j_obs.iter().map(|&x| errorpoly.variance(x)).collect();

                        let sum_k_ijn: f64 = i_obs
                            .iter()
                            .zip(j_obs.iter())
                            .zip(i_var.iter())
                            .zip(j_var.iter())
                            .map(|(((y_i, y_j), i_var), j_var)| {
                                let denominator = i_var + j_var;
                                let term1 = (y_i - y_j).powi(2) / (4.0 * denominator);
                                let term2 = 0.5 * ((i_var + j_var) / 2.0).ln();
                                let term3 = -0.25 * (i_var * j_var).ln();
                                term1 + term2 + term3
                            })
                            .collect::<Vec<f64>>()
                            .iter()
                            .sum::<f64>();

                        let prob_i = predictions.probs[i];
                        let prob_j = predictions.probs[j];
                        let cost = cost_matrix.matrix[(i, j)];
                        let risk_component = prob_i * prob_j * (-sum_k_ijn).exp() * cost;
                        risk += risk_component;
                    }
                }
            }

            (combo.clone(), risk)
        })
        .min_by(|(_, risk_a), (_, risk_b)| risk_a.partial_cmp(risk_b).unwrap())
        .unwrap();

    let res = MmoptResult {
        best_combo_indices: best_combo.clone(),
        best_combo_times: best_combo
            .iter()
            .map(|&index| predictions.times[index])
            .collect(),
        min_risk,
    };

    Ok(res)
}

fn generate_combinations(m: usize, n: usize) -> Vec<Vec<usize>> {
    fn backtrack(
        m: usize,
        n: usize,
        start: usize,
        current: &mut Vec<usize>,
        results: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == n {
            results.push(current.clone());
            return;
        }

        for i in start..m {
            current.push(i);
            backtrack(m, n, i + 1, current, results);
            current.pop();
        }
    }

    let mut results = Vec::new();
    let mut current = Vec::new();
    backtrack(m, n, 0, &mut current, &mut results);
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combinations() {
        let m = 5;
        let n = 3;
        let combinations = generate_combinations(m, n);
        assert_eq!(combinations.len(), 10);
        assert_eq!(combinations[0], vec![0, 1, 2]);
        assert_eq!(combinations[1], vec![0, 1, 3]);
        assert_eq!(combinations[2], vec![0, 1, 4]);
        assert_eq!(combinations[3], vec![0, 2, 3]);
        assert_eq!(combinations[4], vec![0, 2, 4]);
        assert_eq!(combinations[5], vec![0, 3, 4]);
        assert_eq!(combinations[6], vec![1, 2, 3]);
        assert_eq!(combinations[7], vec![1, 2, 4]);
        assert_eq!(combinations[8], vec![1, 3, 4]);
        assert_eq!(combinations[9], vec![2, 3, 4]);
    }
}
