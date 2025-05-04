/*use ndarray::{Array2};

pub mod cross_xxx;
pub mod cross_optimized;

use cross_optimized::{cross_approx, MatrixAccess};

// Адаптер для Array2
pub struct Array2Access<'a> {
    pub matrix: &'a Array2<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::{Uniform, Normal};
    use std::time::Instant;

    // Тест для плотной (dense) случайной матрицы
    #[test]
    fn test_cross_approx_dense_matrix() {
        let n = 500;
        let m = 1000;

        // Создаем плотную случайную матрицу с нормальным распределением
        println!("Creating dense random matrix {}x{}...", n, m);
        let matrix = Array2::<f64>::random((n, m), Normal::new(0.0, 1.0).unwrap());

        // Создаем структуру-адаптер для доступа к матрице
        struct DenseMatrix<'a> {
            data: &'a Array2<f64>,
            rows: usize,
            cols: usize,
        }

        impl<'a> MatrixAccess for DenseMatrix<'a> {
            #[inline(always)]
            fn get(&self, row: usize, col: usize) -> f64 {
                self.data[[row, col]]
            }

            #[inline(always)]
            fn rows(&self) -> usize {
                self.rows
            }

            #[inline(always)]
            fn cols(&self) -> usize {
                self.cols
            }
        }

        let dense_matrix = DenseMatrix {
            data: &matrix,
            rows: n,
            cols: m,
        };

        // Тестируем с разными значениями точности
        for eps in [1e-2, 1e-4, 1e-6, 1e-8].iter() {
            println!("Testing dense matrix approximation with epsilon = {}", eps);

            // Засекаем время
            let start = Instant::now();

            // Вычисляем кросс-аппроксимацию
            let (q, r_mat, rank) = cross_approx(&dense_matrix, None, Some(*eps));

            // Подсчитываем затраченное время
            let duration = start.elapsed();

            // Вычисляем аппроксимацию
            let mut approx = Array2::<f64>::zeros((n, m));
            for i in 0..n {
                for j in 0..m {
                    let mut sum = 0.0;
                    for k in 0..rank {
                        sum += q.get(i, k) * r_mat.get(k, j);
                    }
                    approx[[i, j]] = sum;
                }
            }

            // Подсчитываем ошибку аппроксимации
            let mut error_sum = 0.0;
            let mut matrix_norm = 0.0;
            for i in 0..n {
                for j in 0..m {
                    let exact = matrix[[i, j]];
                    let diff = exact - approx[[i, j]];
                    error_sum += diff * diff;
                    matrix_norm += exact * exact;
                }
            }

            let relative_error = (error_sum / matrix_norm).sqrt();
            println!("Dense matrix approximation with rank {}: relative error = {}, time = {:?}",
                     rank, relative_error, duration);
        }
    }

    // Тест для матрицы с экспоненциальным убыванием
    #[test]
    fn test_cross_approx_exponential_matrix() {
        let n = 500;
        let m = 1000;
        let decay_rate = 0.1; // Параметр скорости убывания

        // Создаем структуру для матрицы с экспоненциальным убыванием
        struct ExponentialMatrix {
            rows: usize,
            cols: usize,
            decay_rate: f64,
        }

        impl MatrixAccess for ExponentialMatrix {
            #[inline(always)]
            fn get(&self, row: usize, col: usize) -> f64 {
                let distance = ((row as f64) + (col as f64)).sqrt();
                (-self.decay_rate * distance).exp()
            }

            #[inline(always)]
            fn rows(&self) -> usize {
                self.rows
            }

            #[inline(always)]
            fn cols(&self) -> usize {
                self.cols
            }
        }

        println!("Creating exponential decay matrix {}x{} with decay_rate = {}...",
                 n, m, decay_rate);

        let exp_matrix = ExponentialMatrix {
            rows: n,
            cols: m,
            decay_rate,
        };

        // Создаем полную матрицу для проверки ошибки
        let mut full_matrix = Array2::<f64>::zeros((n, m));
        for i in 0..n {
            for j in 0..m {
                full_matrix[[i, j]] = exp_matrix.get(i, j);
            }
        }

        // Тестируем с разными значениями точности
        for eps in [1e-2, 1e-4, 1e-6, 1e-8].iter() {
            println!("Testing exponential matrix approximation with epsilon = {}", eps);

            // Засекаем время
            let start = Instant::now();

            // Вычисляем кросс-аппроксимацию
            let (q, r_mat, rank) = cross_approx(&exp_matrix, None, Some(*eps));

            // Подсчитываем затраченное время
            let duration = start.elapsed();

            // Вычисляем аппроксимацию
            let mut approx = Array2::<f64>::zeros((n, m));
            for i in 0..n {
                for j in 0..m {
                    let mut sum = 0.0;
                    for k in 0..rank {
                        sum += q.get(i, k) * r_mat.get(k, j);
                    }
                    approx[[i, j]] = sum;
                }
            }

            // Подсчитываем ошибку аппроксимации
            let mut error_sum = 0.0;
            let mut matrix_norm = 0.0;
            for i in 0..n {
                for j in 0..m {
                    let exact = full_matrix[[i, j]];
                    let diff = exact - approx[[i, j]];
                    error_sum += diff * diff;
                    matrix_norm += exact * exact;
                }
            }

            let relative_error = (error_sum / matrix_norm).sqrt();
            println!("Exponential matrix approximation with rank {}: relative error = {}, time = {:?}",
                     rank, relative_error, duration);
        }
    }

    // Тест для матрицы низкого ранга
    #[test]
    fn test_cross_approx_low_rank_matrix() {
        let n = 500;
        let m = 1000;
        let true_rank = 20; // Истинный ранг матрицы

        println!("Creating low rank matrix {}x{} with rank = {}...",
                 n, m, true_rank);

        // Создаем матрицу низкого ранга как произведение двух случайных матриц
        let left = Array2::<f64>::random((n, true_rank), Normal::new(0.0, 1.0).unwrap());
        let right = Array2::<f64>::random((true_rank, m), Normal::new(0.0, 1.0).unwrap());

        // Структура для эффективного доступа к матрице низкого ранга
        struct LowRankMatrix {
            left: Array2<f64>,
            right: Array2<f64>,
            rows: usize,
            cols: usize,
        }

        impl MatrixAccess for LowRankMatrix {
            #[inline(always)]
            fn get(&self, row: usize, col: usize) -> f64 {
                let mut sum = 0.0;
                for k in 0..self.left.shape()[1] {
                    sum += self.left[[row, k]] * self.right[[k, col]];
                }
                sum
            }

            #[inline(always)]
            fn rows(&self) -> usize {
                self.rows
            }

            #[inline(always)]
            fn cols(&self) -> usize {
                self.cols
            }
        }

        let low_rank_matrix = LowRankMatrix {
            left,
            right,
            rows: n,
            cols: m,
        };

        // Создаем полную матрицу для проверки ошибки
        let mut full_matrix = Array2::<f64>::zeros((n, m));
        for i in 0..n {
            for j in 0..m {
                full_matrix[[i, j]] = low_rank_matrix.get(i, j);
            }
        }

        // Тестируем с разными значениями точности
        for eps in [1e-2, 1e-4, 1e-6, 1e-8].iter() {
            println!("Testing low rank matrix approximation with epsilon = {}", eps);

            // Засекаем время
            let start = Instant::now();

            // Вычисляем кросс-аппроксимацию
            let (q, r_mat, rank) = cross_approx(&low_rank_matrix, None, Some(*eps));

            // Подсчитываем затраченное время
            let duration = start.elapsed();

            // Вычисляем аппроксимацию
            let mut approx = Array2::<f64>::zeros((n, m));
            for i in 0..n {
                for j in 0..m {
                    let mut sum = 0.0;
                    for k in 0..rank {
                        sum += q.get(i, k) * r_mat.get(k, j);
                    }
                    approx[[i, j]] = sum;
                }
            }

            // Подсчитываем ошибку аппроксимации
            let mut error_sum = 0.0;
            let mut matrix_norm = 0.0;
            for i in 0..n {
                for j in 0..m {
                    let exact = full_matrix[[i, j]];
                    let diff = exact - approx[[i, j]];
                    error_sum += diff * diff;
                    matrix_norm += exact * exact;
                }
            }

            let relative_error = (error_sum / matrix_norm).sqrt();
            println!("Low rank (true rank = {}) matrix approximation with computed rank {}: relative error = {}, time = {:?}",
                     true_rank, rank, relative_error, duration);

            // Проверяем, что алгоритм действительно находит ранг, близкий к истинному
            if *eps <= 1e-6 {
                println!("Checking if computed rank ({}) is close to true rank ({})", rank, true_rank);
                assert!(rank >= true_rank - 5 && rank <= true_rank + 5,
                        "Computed rank {} is too far from true rank {}", rank, true_rank);
            }
        }
    }

    // Сравнительный тест для всех типов матриц
    #[test]
    fn test_compare_matrix_types() {
        let n = 300;
        let m = 600;
        let eps = 1e-6;
        let decay_rate = 0.1;
        let true_rank = 15;

        println!("\nComparing different matrix types with size {}x{} and epsilon = {}", n, m, eps);
        println!("{:-<80}", "");
        println!("{:<15} | {:<10} | {:<15} | {:<15} | {:<15}",
                 "Matrix Type", "Rank", "Rel. Error", "Time (ms)", "Mem Usage");
        println!("{:-<80}", "");

        // 1. Матрица Гильберта
        struct HilbertMatrix {
            rows: usize,
            cols: usize,
        }

        impl MatrixAccess for HilbertMatrix {
            fn get(&self, row: usize, col: usize) -> f64 {
                1.0 / ((row + col + 1) as f64)
            }

            fn rows(&self) -> usize {
                self.rows
            }

            fn cols(&self) -> usize {
                self.cols
            }
        }

        let hilbert_matrix = HilbertMatrix { rows: n, cols: m };

        // Засекаем время
        let start = Instant::now();

        // Вычисляем кросс-аппроксимацию
        let (q_h, r_h, rank_h) = cross_approx(&hilbert_matrix, None, Some(eps));

        // Подсчитываем затраченное время
        let duration_h = start.elapsed();

        // Вычисляем полную и аппроксимированную матрицы
        let mut hilbert_full = Array2::<f64>::zeros((n, m));
        let mut hilbert_approx = Array2::<f64>::zeros((n, m));

        for i in 0..n {
            for j in 0..m {
                hilbert_full[[i, j]] = hilbert_matrix.get(i, j);
                let mut sum = 0.0;
                for k in 0..rank_h {
                    sum += q_h.get(i, k) * r_h.get(k, j);
                }
                hilbert_approx[[i, j]] = sum;
            }
        }

        // Подсчитываем ошибку
        let mut error_sum_h = 0.0;
        let mut norm_h = 0.0;

        for i in 0..n {
            for j in 0..m {
                let diff = hilbert_full[[i, j]] - hilbert_approx[[i, j]];
                error_sum_h += diff * diff;
                norm_h += hilbert_full[[i, j]] * hilbert_full[[i, j]];
            }
        }

        let rel_error_h = (error_sum_h / norm_h).sqrt();

        println!("{:<15} | {:<10} | {:<15.2e} | {:<15.2} | {:<15}",
                 "Hilbert", rank_h, rel_error_h, duration_h.as_millis(), "Small");

        // 2. Экспоненциальная матрица
        struct ExponentialMatrix {
            rows: usize,
            cols: usize,
            decay_rate: f64,
        }

        impl MatrixAccess for ExponentialMatrix {
            fn get(&self, row: usize, col: usize) -> f64 {
                let distance = ((row as f64) + (col as f64)).sqrt();
                (-self.decay_rate * distance).exp()
            }

            fn rows(&self) -> usize {
                self.rows
            }

            fn cols(&self) -> usize {
                self.cols
            }
        }

        let exp_matrix = ExponentialMatrix {
            rows: n,
            cols: m,
            decay_rate,
        };

        // Засекаем время
        let start = Instant::now();

        // Вычисляем кросс-аппроксимацию
        let (q_e, r_e, rank_e) = cross_approx(&exp_matrix, None, Some(eps));

        // Подсчитываем затраченное время
        let duration_e = start.elapsed();

        // Вычисляем полную и аппроксимированную матрицы
        let mut exp_full = Array2::<f64>::zeros((n, m));
        let mut exp_approx = Array2::<f64>::zeros((n, m));

        for i in 0..n {
            for j in 0..m {
                exp_full[[i, j]] = exp_matrix.get(i, j);
                let mut sum = 0.0;
                for k in 0..rank_e {
                    sum += q_e.get(i, k) * r_e.get(k, j);
                }
                exp_approx[[i, j]] = sum;
            }
        }

        // Подсчитываем ошибку
        let mut error_sum_e = 0.0;
        let mut norm_e = 0.0;

        for i in 0..n {
            for j in 0..m {
                let diff = exp_full[[i, j]] - exp_approx[[i, j]];
                error_sum_e += diff * diff;
                norm_e += exp_full[[i, j]] * exp_full[[i, j]];
            }
        }

        let rel_error_e = (error_sum_e / norm_e).sqrt();

        println!("{:<15} | {:<10} | {:<15.2e} | {:<15.2} | {:<15}",
                 "Exponential", rank_e, rel_error_e, duration_e.as_millis(), "Small");

        // 3. Матрица низкого ранга
        let left = Array2::<f64>::random((n, true_rank), Normal::new(0.0, 1.0).unwrap());
        let right = Array2::<f64>::random((true_rank, m), Normal::new(0.0, 1.0).unwrap());

        struct LowRankMatrix {
            left: Array2<f64>,
            right: Array2<f64>,
            rows: usize,
            cols: usize,
        }

        impl MatrixAccess for LowRankMatrix {
            fn get(&self, row: usize, col: usize) -> f64 {
                let mut sum = 0.0;
                for k in 0..self.left.shape()[1] {
                    sum += self.left[[row, k]] * self.right[[k, col]];
                }
                sum
            }

            fn rows(&self) -> usize {
                self.rows
            }

            fn cols(&self) -> usize {
                self.cols
            }
        }

        let low_rank_matrix = LowRankMatrix {
            left,
            right,
            rows: n,
            cols: m,
        };

        // Засекаем время
        let start = Instant::now();

        // Вычисляем кросс-аппроксимацию
        let (q_l, r_l, rank_l) = cross_approx(&low_rank_matrix, None, Some(eps));

        // Подсчитываем затраченное время
        let duration_l = start.elapsed();

        // Вычисляем полную и аппроксимированную матрицы
        let mut low_rank_full = Array2::<f64>::zeros((n, m));
        let mut low_rank_approx = Array2::<f64>::zeros((n, m));

        for i in 0..n {
            for j in 0..m {
                low_rank_full[[i, j]] = low_rank_matrix.get(i, j);
                let mut sum = 0.0;
                for k in 0..rank_l {
                    sum += q_l.get(i, k) * r_l.get(k, j);
                }
                low_rank_approx[[i, j]] = sum;
            }
        }

        // Подсчитываем ошибку
        let mut error_sum_l = 0.0;
        let mut norm_l = 0.0;

        for i in 0..n {
            for j in 0..m {
                let diff = low_rank_full[[i, j]] - low_rank_approx[[i, j]];
                error_sum_l += diff * diff;
                norm_l += low_rank_full[[i, j]] * low_rank_full[[i, j]];
            }
        }

        let rel_error_l = (error_sum_l / norm_l).sqrt();

        println!("{:<15} | {:<10} | {:<15.2e} | {:<15.2} | {:<15}",
                 "Low Rank", rank_l, rel_error_l, duration_l.as_millis(),
                 format!("{}KB", (n + m) * true_rank * 8 / 1024));

        // 4. Плотная случайная матрица
        // Для плотной матрицы мы используем небольшую подвыборку из-за памяти
        let n_dense = n / 2;
        let m_dense = m / 2;
        let dense_matrix = Array2::<f64>::random((n_dense, m_dense), Normal::new(0.0, 1.0).unwrap());

        struct DenseMatrix<'a> {
            data: &'a Array2<f64>,
            rows: usize,
            cols: usize,
        }

        impl<'a> MatrixAccess for DenseMatrix<'a> {
            fn get(&self, row: usize, col: usize) -> f64 {
                self.data[[row, col]]
            }

            fn rows(&self) -> usize {
                self.rows
            }

            fn cols(&self) -> usize {
                self.cols
            }
        }

        let dense_matrix_access = DenseMatrix {
            data: &dense_matrix,
            rows: n_dense,
            cols: m_dense,
        };

        // Засекаем время
        let start = Instant::now();

        // Вычисляем кросс-аппроксимацию
        let (q_d, r_d, rank_d) = cross_approx(&dense_matrix_access, None, Some(eps));

        // Подсчитываем затраченное время
        let duration_d = start.elapsed();

        // Вычисляем аппроксимированную матрицу
        let mut dense_approx = Array2::<f64>::zeros((n_dense, m_dense));

        for i in 0..n_dense {
            for j in 0..m_dense {
                let mut sum = 0.0;
                for k in 0..rank_d {
                    sum += q_d.get(i, k) * r_d.get(k, j);
                }
                dense_approx[[i, j]] = sum;
            }
        }

        // Подсчитываем ошибку
        let mut error_sum_d = 0.0;
        let mut norm_d = 0.0;

        for i in 0..n_dense {
            for j in 0..m_dense {
                let diff = dense_matrix[[i, j]] - dense_approx[[i, j]];
                error_sum_d += diff * diff;
                norm_d += dense_matrix[[i, j]] * dense_matrix[[i, j]];
            }
        }

        let rel_error_d = (error_sum_d / norm_d).sqrt();

        println!("{:<15} | {:<10} | {:<15.2e} | {:<15.2} | {:<15}",
                 "Dense Random", rank_d, rel_error_d, duration_d.as_millis(),
                 format!("{}MB", n_dense * m_dense * 8 / (1024*1024)));
        println!("{:-<80}", "");

        // Сравнение и выводы
        println!("\nВыводы из сравнения матриц:");
        println!("1. Наименьший ранг для достижения точности {}: {}", eps,
                 [rank_h, rank_e, rank_l, rank_d].iter().min().unwrap());
        println!("2. Наименьшая относительная ошибка: {:.2e}",
                 [rel_error_h, rel_error_e, rel_error_l, rel_error_d].iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap());
        println!("3. Наименьшее время выполнения: {} мс",
                 [duration_h.as_millis(), duration_e.as_millis(), duration_l.as_millis(), duration_d.as_millis()].iter().min().unwrap());
    }

    #[test]
    fn test_simple_dense_matrix_comparison() {
        use super::*;
        use ndarray::Array2;
        use ndarray_rand::RandomExt;
        use ndarray_rand::rand_distr::Uniform;
        use std::time::Instant;

        // Создаем матрицу небольшого размера для наглядности
        let n = 10;
        let m = 15;

        println!("\n### Простое сравнение исходной dense матрицы и её аппроксимации ###");
        println!("Размер матрицы: {}x{}", n, m);

        // Создаем плотную матрицу со значениями от 0 до 10 для лучшей читаемости
        let matrix = Array2::<f64>::random((n, m), Uniform::new(0.0, 10.0));

        // Структура для доступа к матрице через MatrixAccess
        #[derive(Debug)]
        struct DenseMatrix<'a> {
            data: &'a Array2<f64>,
        }

        impl<'a> MatrixAccess for DenseMatrix<'a> {
            #[inline(always)]
            fn get(&self, row: usize, col: usize) -> f64 {
                self.data[[row, col]]
            }

            #[inline(always)]
            fn rows(&self) -> usize {
                self.data.shape()[0]
            }

            #[inline(always)]
            fn cols(&self) -> usize {
                self.data.shape()[1]
            }
        }

        let dense_matrix = DenseMatrix { data: &matrix };

        // Запускаем алгоритм с фиксированным рангом = 3 (небольшой ранг для наглядности)
        let rank = 10;
        let start = Instant::now();
        let (q, r, actual_rank) = cross_approx(&dense_matrix, Some(rank), Some(1e-10));
        let elapsed = start.elapsed();

        println!("\nРезультаты кросс-аппроксимации:");
        println!("Запрошенный ранг: {}", rank);
        println!("Фактический ранг: {}", actual_rank);
        println!("Время выполнения: {:.2?}", elapsed);

        // Вычисляем аппроксимацию A ≈ Q·R
        let mut approx = Array2::<f64>::zeros((n, m));
        for i in 0..n {
            for j in 0..m {
                let mut sum = 0.0;
                for k in 0..actual_rank {
                    sum += q.get(i, k) * r.get(k, j);
                }
                approx[[i, j]] = sum;
            }
        }
        println!("{:.2?}", approx);
        println!("{:.2?}", dense_matrix);
        // Вычисляем ошибку
        let mut error_sum = 0.0;
        let mut matrix_sum = 0.0;
        for i in 0..n {
            for j in 0..m {
                let diff = matrix[[i, j]] - approx[[i, j]];
                error_sum += diff * diff;
                matrix_sum += matrix[[i, j]] * matrix[[i, j]];
            }
        }
        let relative_error = (error_sum / matrix_sum).sqrt();

        println!("Относительная ошибка: {:.6e}", relative_error);

        // Выводим сравнение исходной матрицы и аппроксимации
        println!("\n1. Исходная матрица (первые 5x5 элементов):");
        print_submatrix(&matrix, 5, 5);

        println!("\n2. Аппроксимация ранга {} (первые 5x5 элементов):", actual_rank);
        print_submatrix(&approx, 5, 5);

        println!("\n3. Абсолютная ошибка |A - QR| (первые 5x5 элементов):");
        print_error_matrix(&matrix, &approx, 5, 5);

        // Проверяем, что произведение Q*R действительно близко к исходной матрице
        let max_allowed_error = 1e-2; // Допустимая ошибка для маленького ранга
        let max_error = get_max_abs_error(&matrix, &approx);

        println!("\nМаксимальная абсолютная ошибка: {:.6}", max_error);
        println!("Среднеквадратическая ошибка: {:.6}", (error_sum / (n * m) as f64).sqrt());

        if relative_error <= max_allowed_error {
            println!("\nТест ПРОЙДЕН: Относительная ошибка в пределах допустимой ({:.6e} <= {:.6e})",
                     relative_error, max_allowed_error);
        } else {
            println!("\nТест НЕ ПРОЙДЕН: Относительная ошибка превышает допустимую ({:.6e} > {:.6e})",
                     relative_error, max_allowed_error);
        }
    }

    // Вспомогательная функция для вывода подматрицы
    fn print_submatrix(matrix: &Array2<f64>, rows: usize, cols: usize) {
        let actual_rows = std::cmp::min(rows, matrix.shape()[0]);
        let actual_cols = std::cmp::min(cols, matrix.shape()[1]);

        for i in 0..actual_rows {
            for j in 0..actual_cols {
                print!("{:6.2} ", matrix[[i, j]]);
            }
            println!();
        }
    }

    // Вспомогательная функция для вывода матрицы ошибки
    fn print_error_matrix(original: &Array2<f64>, approx: &Array2<f64>, rows: usize, cols: usize) {
        let actual_rows = std::cmp::min(rows, original.shape()[0]);
        let actual_cols = std::cmp::min(cols, original.shape()[1]);

        for i in 0..actual_rows {
            for j in 0..actual_cols {
                let error = (original[[i, j]] - approx[[i, j]]).abs();
                print!("{:6.2} ", error);
            }
            println!();
        }
    }

    // Вспомогательная функция для нахождения максимальной абсолютной ошибки
    fn get_max_abs_error(original: &Array2<f64>, approx: &Array2<f64>) -> f64 {
        let mut max_error = 0.0;

        for i in 0..original.shape()[0] {
            for j in 0..original.shape()[1] {
                let error = (original[[i, j]] - approx[[i, j]]).abs();
                if error > max_error {
                    max_error = error;
                }
            }
        }

        max_error
    }

}*/
use std::time::{Instant, Duration};
use std::mem::size_of;
use ndarray::{Array1, ArrayView1};

pub mod cross_optimized;
// Импортируем необходимые структуры для работы с ChunkedMatrix
use cross_optimized::ChunkedMatrix;

// Импортируем необходимые структуры из основного модуля
use cross_optimized::{MatrixAccess, cross_approx, estimate_cross_approx_memory};

// Реализация экспоненциальной матрицы как ленивой структуры данных
// Вычисляет элементы на лету по формуле exp(-|i-j|/sigma)
struct ExpMatrix {
    rows: usize,
    cols: usize,
    sigma: f64,
}

impl ExpMatrix {
    fn new(rows: usize, cols: usize, sigma: f64) -> Self {
        Self { rows, cols, sigma }
    }
}

// Реализация трейта MatrixAccess для экспоненциальной матрицы
impl MatrixAccess for ExpMatrix {
    #[inline(always)]
    fn get(&self, row: usize, col: usize) -> f64 {
        debug_assert!(row < self.rows && col < self.cols, "Index out of bounds");

        // Формула экспоненциальной матрицы: exp(-|i-j|/sigma)
        let diff = if row > col { row - col } else { col - row };
        (-(diff as f64) / self.sigma).exp()
    }

    #[inline(always)]
    fn rows(&self) -> usize {
        self.rows
    }

    #[inline(always)]
    fn cols(&self) -> usize {
        self.cols
    }
}

// Структура для отслеживания выделения памяти
struct MemoryTracker {
    start_allocated: usize,
    peak_allocated: usize,
    current_allocated: usize,
}

impl MemoryTracker {
    fn new() -> Self {
        // В реальном коде здесь нужно использовать подходящую библиотеку для отслеживания памяти
        // Например, jemalloc с профилированием или библиотеку memory_stats
        // В этом примере мы просто симулируем это
        let current = get_current_memory_usage();
        Self {
            start_allocated: current,
            peak_allocated: current,
            current_allocated: current,
        }
    }

    fn update(&mut self) {
        self.current_allocated = get_current_memory_usage();
        if self.current_allocated > self.peak_allocated {
            self.peak_allocated = self.current_allocated;
        }
    }

    fn report(&self) -> (usize, usize) {
        // Возвращает (пиковое использование, текущее использование) в байтах
        (
            self.peak_allocated - self.start_allocated,
            self.current_allocated - self.start_allocated
        )
    }
}

// Вспомогательная функция для получения текущего использования памяти
// В реальном коде это должно использовать системные API или библиотеки
fn get_current_memory_usage() -> usize {
    // В Linux можно использовать /proc/self/statm
    #[cfg(target_os = "linux")]
    {
        use std::fs::File;
        use std::io::Read;

        let mut buffer = String::new();
        if let Ok(mut file) = File::open("/proc/self/statm") {
            if file.read_to_string(&mut buffer).is_ok() {
                if let Some(value) = buffer.split_whitespace().next() {
                    if let Ok(pages) = value.parse::<usize>() {
                        // В Linux размер страницы обычно 4KB
                        return pages * 4 * 1024;
                    }
                }
            }
        }
    }

    // В Windows можно использовать GetProcessMemoryInfo через winapi
    #[cfg(target_os = "windows")]
    {
        // В реальном коде здесь бы использовался winapi
        // Примерно так:
        // use winapi::um::processthreadsapi::GetCurrentProcess;
        // use winapi::um::psapi::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    }

    // Если нет доступа к системным API или произошла ошибка,
    // используем простое приближение на основе выделенных объектов в тесте
    // Это неточная оценка, но позволит увидеть относительные изменения

    // В реальном коде лучше использовать библиотеки:
    // - jemalloc с профилированием
    // - crate memory-stats (https://crates.io/crates/memory-stats)
    // - psutil-rs для кроссплатформенной поддержки

    // Возвращаем заглушку (1MB) в качестве базовой метрики
    // В реальном приложении замените на реальный вызов библиотеки
    1024 * 1024
}

// Функция тестирования производительности cross_approx
fn benchmark_cross_approx() {
    // Параметры теста
    let n = 1_000_000; // Размерность матрицы (строки)
    let m = 1_000_000; // Размерность матрицы (столбцы)
    let sigma = 100.0; // Параметр экспоненциальной матрицы
    let rank = 100;    // Ранг аппроксимации

    println!("Начало теста производительности cross_approx");
    println!("Размерность матрицы: {}x{}", n, m);
    println!("Параметр сигма: {}", sigma);
    println!("Ранк аппроксимации: {}", rank);

    // Создаём экспоненциальную матрицу (ленивую, не храним всю)
    let matrix = ExpMatrix::new(n, m, sigma);

    // Оценка требуемой памяти
    let estimated_memory = estimate_cross_approx_memory(&matrix, Some(rank));
    println!("Оценка требуемой памяти: {:.2} ГБ", estimated_memory as f64 / (1024.0 * 1024.0 * 1024.0));

    // Инициализируем трекер памяти
    let mut memory_tracker = MemoryTracker::new();

    // Измеряем время выполнения
    let start_time = Instant::now();

    // Запускаем алгоритм кросс-аппроксимации
    let (q_matrix, r_matrix, actual_rank) = cross_approx(&matrix, None, Some(1e-6));

    let elapsed = start_time.elapsed();

    // Обновляем информацию о памяти
    memory_tracker.update();
    let (peak_memory, current_memory) = memory_tracker.report();

    // Выводим результаты
    println!("Время выполнения: {:.2} сек", elapsed.as_secs_f64());
    println!("Достигнутый ранг: {}", actual_rank);
    println!("Пиковое использование памяти: {:.2} ГБ", peak_memory as f64 / (1024.0 * 1024.0 * 1024.0));
    println!("Текущее использование памяти: {:.2} ГБ", current_memory as f64 / (1024.0 * 1024.0 * 1024.0));

    // Проверка точности аппроксимации
    validate_approximation(&matrix, &q_matrix, &r_matrix, actual_rank);
}

// Функция для проверки качества аппроксимации
fn validate_approximation<M: MatrixAccess>(
    original: &M,
    q_matrix: &ChunkedMatrix,
    r_matrix: &ChunkedMatrix,
    rank: usize
) {
    // Выберем несколько случайных точек для проверки
    let num_samples = 10;
    let mut total_error = 0.0;

    // Используем фиксированные семена для воспроизводимости
    let sample_points = [
        (1000, 2000),
        (50000, 60000),
        (500000, 400000),
        (900000, 800000),
        (100, 999999),
        (999999, 100),
        (400000, 400000),
        (700000, 300000),
        (123456, 654321),
        (777777, 888888),
    ];

    for &(i, j) in &sample_points[0..num_samples] {
        if i < original.rows() && j < original.cols() {
            let original_value = original.get(i, j);

            // Вычисляем аппроксимацию
            let mut approx_value = 0.0;
            for k in 0..rank {
                approx_value += q_matrix.get(i, k) * r_matrix.get(k, j);
            }

            let error = (original_value - approx_value).abs();
            total_error += error;

            println!("Элемент [{}, {}]: оригинал = {:.6e}, аппроксимация = {:.6e}, ошибка = {:.6e}",
                     i, j, original_value, approx_value, error);
        }
    }

    println!("Средняя абсолютная ошибка по {} точкам: {:.6e}", num_samples, total_error / num_samples as f64);
}

#[cfg(test)]
mod tests {
    use super::*;

    // Тест производительности кросс-аппроксимации
    #[test]
    //#[ignore] // Игнорируем по умолчанию из-за длительного выполнения
    fn test_cross_approx_performance() {
        // Отключаем логирование для чистоты замеров производительности
        std::env::set_var("RUST_LOG", "error");

        // Запускаем тест
        benchmark_cross_approx();
    }
}

// Если хотим запустить как отдельное приложение
fn main() {
    // Инициализируем логгер (требуется добавить env_logger в dependencies)
    #[cfg(feature = "logging")]
    env_logger::init();

    // Запускаем тест производительности
    benchmark_cross_approx();
}