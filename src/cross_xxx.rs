// cross_approx.rs — финальная оптимизированная версия

use ndarray::{Array2, ArrayView2, ArrayView1, Axis, s};
use std::cmp::min;
use rand::{thread_rng, Rng};
use ndarray_rand::rand_distr::Uniform;

// Находим координаты максимального по модулю элемента
fn argmax_abs(arr: ArrayView2<f64>) -> (usize, usize) {
    arr.indexed_iter()
        .max_by(|(_, &a), (_, &b)| a.abs().partial_cmp(&b.abs()).unwrap())
        .map(|((i, j), _)| (i, j))
        .unwrap_or((0, 0))
}

// Вычисляем норму Фробениуса
fn calculate_norm(matrix: &Array2<f64>) -> f64 {
    matrix.iter().map(|x| x.powi(2)).sum()
}

// Структура для динамического роста матриц
struct DynamicMatrix {
    data: Array2<f64>,
    axis: Axis,
    size: usize,
    block: usize,
}

impl DynamicMatrix {
    fn new(dim_other: usize, axis: Axis, block: usize) -> Self {
        let shape = match axis {
            Axis(0) => (block, dim_other),
            Axis(1) => (dim_other, block),
            _ => panic!("Unsupported axis"),
        };
        DynamicMatrix {
            data: Array2::<f64>::zeros(shape),
            axis,
            size: 0,
            block,
        }
    }

    fn push(&mut self, vector: ArrayView1<f64>) {
        if self.size >= self.data.len_of(self.axis) {
            self.grow();
        }
        match self.axis {
            Axis(0) => self.data.row_mut(self.size).assign(&vector),
            Axis(1) => self.data.column_mut(self.size).assign(&vector),
            _ => unreachable!(),
        }
        self.size += 1;
    }

    fn grow(&mut self) {
        let (rows, cols) = self.data.dim();
        let (new_rows, new_cols) = match self.axis {
            Axis(0) => (rows + self.block, cols),
            Axis(1) => (rows, cols + self.block),
            _ => panic!("Unsupported axis"),
        };
        let mut new_data = Array2::<f64>::zeros((new_rows, new_cols));
        new_data.slice_mut(s![..rows, ..cols]).assign(&self.data);
        self.data = new_data;
    }

    fn view(&self) -> Array2<f64> {
        match self.axis {
            Axis(0) => self.data.slice(s![..self.size, ..]).to_owned(),
            Axis(1) => self.data.slice(s![.., ..self.size]).to_owned(),
            _ => panic!("Unsupported axis"),
        }
    }
}

// Основная функция крестового разложения
pub fn cross_approx(
    matrix: ArrayView2<f64>,
    r: Option<usize>,
    _eps: Option<f64>,
) -> (Array2<f64>, Array2<f64>, usize) {
    let (n, m) = (matrix.nrows(), matrix.ncols());
    let target_r = r.unwrap_or_else(|| min(n, m));

    let mut residual = matrix.to_owned();
    let mut q_dyn = DynamicMatrix::new(n, Axis(1), 16);
    let mut r_dyn = DynamicMatrix::new(m, Axis(0), 16);

    let mut rank = 0;

    while rank < target_r {
        let (i_max, j_max) = argmax_abs(residual.view());
        let pivot = residual[[i_max, j_max]];

        if pivot.abs() < 1e-12 {
            break;
        }

        let mut q_col = residual.column(j_max).to_owned();
        q_col.mapv_inplace(|x| x / pivot);

        let r_row = residual.row(i_max).to_owned();

        let a_max_q = q_col.iter().map(|x| x.abs()).fold(0.0, f64::max);
        let a_max_r = r_row.iter().map(|x| x.abs()).fold(0.0, f64::max);
        let a_max_step = a_max_q * a_max_r;

        if a_max_step.sqrt() * (n as f64).sqrt() * (m as f64).sqrt() <= pivot.abs() {
            break;
        }

        q_dyn.push(q_col.view());
        r_dyn.push(r_row.view());

        for (i, q_val) in q_col.iter().enumerate() {
            for (j, r_val) in r_row.iter().enumerate() {
                residual[[i, j]] -= q_val * r_val;
            }
        }

        rank += 1;
    }

    (q_dyn.view(), r_dyn.view(), rank)
}
