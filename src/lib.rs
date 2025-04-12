pub mod cross;

#[cfg(test)]
mod tests {
    use ndarray::{s, Array2};
    use crate::cross::cross_approx;

    #[test]
    fn test_cross_approximation() {
        let n = 10;
        let m = 8;
        let matrix = Array2::<f64>::from_shape_fn((n, m), |(i, j)| (i + j) as f64);

        let (q, r, rank) = cross_approx(matrix.view(), Some(2), None);

        assert_eq!(q.shape(), &[n, 2]);
        assert_eq!(r.shape(), &[2, m]);
        assert_eq!(rank, 2);

        let approximated = q.slice(s![.., ..rank]).dot(&r.slice(s![..rank, ..]));

        let error = (&matrix - &approximated).mapv(|x| x.powi(2)).sum().sqrt();
        let relative_error = error / matrix.mapv(|x| x.powi(2)).sum().sqrt();

        assert!(relative_error < 0.3, "Relative error too high: {}", relative_error);
    }

    #[test]
    fn test_rank_one_approximation() {
        let n = 5;
        let m = 4;
        let matrix = Array2::<f64>::from_shape_fn((n, m), |(i, j)| (i + j) as f64);

        let (q, r, rank) = cross_approx(matrix.view(), Some(1), None);

        assert_eq!(q.shape(), &[n, 1]);
        assert_eq!(r.shape(), &[1, m]);
        assert_eq!(rank, 1);
    }

    #[test]
    fn test_cauchy_matrix() {
        let n = 20;
        let m = 15;
        let matrix = Array2::<f64>::from_shape_fn((n, m), |(i, j)| 1.0 / ((i + j + 1) as f64));

        for test_rank in &[1, 2, 4, 8] {
            let rank = *test_rank;
            let (q, r, actual_rank) = cross_approx(matrix.view(), Some(rank), None);

            assert_eq!(q.shape(), &[n, rank]);
            assert_eq!(r.shape(), &[rank, m]);
            assert_eq!(actual_rank, rank);

            let approximated = q.slice(s![.., ..actual_rank]).dot(&r.slice(s![..actual_rank, ..]));

            let error = (&matrix - &approximated).mapv(|x| x.powi(2)).sum().sqrt();
            let relative_error = error / matrix.mapv(|x| x.powi(2)).sum().sqrt();

            println!("Cauchy matrix, rank {}: relative error = {:.6}", rank, relative_error);

            let expected_max_error = match rank {
                1 => 0.4,
                2 => 0.2,
                4 => 0.05,
                8 => 0.01,
                _ => 0.5,
            };

            assert!(
                relative_error < expected_max_error,
                "Rank-{} approximation of Cauchy matrix has too high error: {:.6} (expected < {:.6})",
                rank, relative_error, expected_max_error
            );
        }
    }
}