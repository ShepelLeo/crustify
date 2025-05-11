use ndarray::{Array1, ArrayView1};
use rand::{thread_rng, Rng};
use std::cmp::{max, min};

const BOUND: usize = 1_000_000;

pub trait MatrixAccess {
    fn get(&self, row: usize, col: usize) -> f64;

    fn rows(&self) -> usize;

    fn cols(&self) -> usize;
}

pub struct ChunkedMatrix {
    chunks: Vec<Vec<f64>>,
    rows: usize,
    cols: usize,
    alloc_rows: usize,
    alloc_cols: usize,
    actual_cols_used: usize,
}

impl ChunkedMatrix {
    fn new(initial_rows: usize, initial_cols: usize, max_rows: usize, max_cols: usize) -> Self {
        let init_rows = 16.max(initial_rows.min(max_rows));
        let init_cols = 16.max(initial_cols.min(max_cols));

        let mut chunks = Vec::with_capacity(1 + (max_rows / init_rows));
        chunks.push(vec![0.0; init_rows * init_cols]);

        Self {
            chunks,
            rows: 0,
            cols: initial_cols,
            alloc_rows: init_rows,
            alloc_cols: init_cols,
            actual_cols_used: initial_cols,
        }
    }

    fn zeros(rows: usize, cols: usize) -> Self {
        let mut matrix = Self::new(rows, cols, rows, cols);
        matrix.rows = rows;
        matrix.actual_cols_used = cols;
        matrix
    }

    #[inline(always)]
    fn get(&self, row: usize, col: usize) -> f64 {
        debug_assert!(row < self.rows && col < self.cols, "Index out of bounds");

        let chunk_idx = row / self.alloc_rows;
        let row_within_chunk = row % self.alloc_rows;

        if chunk_idx < self.chunks.len() {
            self.chunks[chunk_idx][row_within_chunk * self.alloc_cols + col]
        } else {
            0.0
        }
    }

    #[inline(always)]
    fn ensure_chunk_available(&mut self, chunk_idx: usize) -> bool {
        let mut added = false;
        while chunk_idx >= self.chunks.len() {
            if self.chunks.len() * self.alloc_rows >= BOUND {
                panic!("Matrix has too many rows: {}", self.chunks.len() * self.alloc_rows);
            }
            self.chunks.push(vec![0.0; self.alloc_rows * self.alloc_cols]);
            added = true;
        }
        added
    }

    #[inline(always)]
    fn ensure_chunks_count(&mut self, needed_chunks: usize) -> bool {
        let mut added = false;
        while needed_chunks > self.chunks.len() {
            if self.chunks.len() * self.alloc_rows >= BOUND {
                panic!("Matrix has too many rows: {}", self.chunks.len() * self.alloc_rows);
            }
            self.chunks.push(vec![0.0; self.alloc_rows * self.alloc_cols]);
            added = true;
        }
        added
    }

    #[inline(always)]
    fn set(&mut self, row: usize, col: usize, value: f64) {
        if row >= BOUND || col >= BOUND {
            panic!("Matrix dimensions too large: {}x{}", row, col);
        }

        if row >= self.rows {
            self.rows = row + 1;
        }

        if col >= self.alloc_cols {
            self.expand_cols(col + 1);
        }

        if col >= self.actual_cols_used {
            self.actual_cols_used = col + 1;
        }

        let chunk_idx = row / self.alloc_rows;
        let row_within_chunk = row % self.alloc_rows;

        self.ensure_chunk_available(chunk_idx);

        self.chunks[chunk_idx][row_within_chunk * self.alloc_cols + col] = value;
    }

    fn column(&self, col: usize) -> Array1<f64> {
        debug_assert!(col < self.cols, "Column index out of bounds");

        let mut result = Array1::zeros(self.rows);
        for row in 0..self.rows {
            result[row] = self.get(row, col);
        }
        result
    }

    fn row(&self, row: usize) -> Array1<f64> {
        debug_assert!(row < self.rows, "Row index out of bounds");

        let mut result = Array1::zeros(self.cols);
        for col in 0..self.cols {
            result[col] = self.get(row, col);
        }
        result
    }

    fn set_column(&mut self, col: usize, values: &Array1<f64>) {
        debug_assert!(col < self.alloc_cols, "Column index out of bounds");

        if values.len() > BOUND {
            panic!("Column length too large: {}", values.len());
        }

        if values.len() > self.rows {
            self.rows = values.len();
        }

        if col >= self.actual_cols_used {
            self.actual_cols_used = col + 1;
        }

        let needed_chunks = (values.len() + self.alloc_rows - 1) / self.alloc_rows;
        self.ensure_chunks_count(needed_chunks);

        for (i, &val) in values.iter().enumerate() {
            let chunk_idx = i / self.alloc_rows;
            let row_within_chunk = i % self.alloc_rows;
            self.chunks[chunk_idx][row_within_chunk * self.alloc_cols + col] = val;
        }
    }

    fn set_row(&mut self, row: usize, values: &Array1<f64>) {
        debug_assert!(values.len() <= self.alloc_cols, "Too many values for row");

        if values.len() > BOUND {
            panic!("Row length too large: {}", values.len());
        }

        if row >= self.rows {
            self.rows = row + 1;
        }

        if values.len() > self.actual_cols_used {
            self.actual_cols_used = values.len();
        }

        let chunk_idx = row / self.alloc_rows;
        let row_within_chunk = row % self.alloc_rows;

        self.ensure_chunk_available(chunk_idx);

        for (col, &val) in values.iter().enumerate() {
            self.chunks[chunk_idx][row_within_chunk * self.alloc_cols + col] = val;
        }
    }

    fn expand_cols(&mut self, new_cols: usize) {
        if new_cols <= self.alloc_cols {
            self.cols = new_cols;
            return;
        }

        if new_cols >= BOUND {
            panic!("Column count too large: {}", new_cols);
        }

        let growth_factor = 1.5;
        let max_growth = 10000;

        let additional_cols_needed = new_cols - self.alloc_cols;
        let growth = min(
            (self.alloc_cols as f64 * (growth_factor - 1.0)) as usize,
            max_growth
        );

        let new_alloc_cols = self.alloc_cols + max(additional_cols_needed, growth);

        let mut new_chunks = Vec::with_capacity(self.chunks.len());

        for chunk in &self.chunks {
            let mut new_chunk = vec![0.0; self.alloc_rows * new_alloc_cols];

            for row in 0..self.alloc_rows {
                for col in 0..self.actual_cols_used {
                    if col < self.alloc_cols {
                        new_chunk[row * new_alloc_cols + col] = chunk[row * self.alloc_cols + col];
                    }
                }
            }

            new_chunks.push(new_chunk);
        }

        self.chunks = new_chunks;
        self.alloc_cols = new_alloc_cols;
        self.cols = new_cols;
    }

    fn shrink_to_fit(&mut self) {
        if self.actual_cols_used < self.alloc_cols {
            let new_alloc_cols = self.actual_cols_used;

            let mut new_chunks = Vec::with_capacity(self.chunks.len());

            for chunk in &self.chunks {
                let mut new_chunk = vec![0.0; self.alloc_rows * new_alloc_cols];

                for row in 0..self.alloc_rows {
                    for col in 0..self.actual_cols_used {
                        new_chunk[row * new_alloc_cols + col] = chunk[row * self.alloc_cols + col];
                    }
                }

                new_chunks.push(new_chunk);
            }

            self.chunks = new_chunks;
            self.alloc_cols = new_alloc_cols;
        }

        let needed_chunks = (self.rows + self.alloc_rows - 1) / self.alloc_rows;
        if needed_chunks < self.chunks.len() {
            self.chunks.truncate(needed_chunks);
        }
    }

    fn slice_column_into(&self, col: usize, target: &mut Array1<f64>) {
        debug_assert!(col < self.cols, "Column index out of bounds");
        debug_assert!(target.len() >= self.rows, "Target array too small");

        for row in 0..self.rows {
            target[row] = self.get(row, col);
        }
    }

    fn slice_row_into(&self, row: usize, target: &mut Array1<f64>) {
        debug_assert!(row < self.rows, "Row index out of bounds");
        debug_assert!(target.len() >= self.cols, "Target array too small");

        for col in 0..self.cols {
            target[col] = self.get(row, col);
        }
    }

}

// Implement MatrixAccess for our ChunkedMatrix
impl MatrixAccess for ChunkedMatrix {
    #[inline(always)]
    fn get(&self, row: usize, col: usize) -> f64 {
        self.get(row, col)
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

fn safe_transform_indices<T, F>(range: usize, indices: &[usize], source_len: usize, mapper: F) -> Vec<T>
where
    T: Copy + Default,
    F: Fn(usize) -> T,
{
    (0..range)
        .filter_map(|j| {
            let idx = indices.get(j)?;
            if *idx < source_len {
                Some(mapper(*idx))
            } else {
                Some(T::default())
            }
        })
        .collect()
}

fn shift_columns_left(arr: &mut ChunkedMatrix, col_idx: usize, limit: usize) {
    let n = arr.rows;

    for j in col_idx..limit-1 {
        for i in 0..n {
            let next_col_val = arr.get(i, j+1);
            arr.set(i, j, next_col_val);
        }
    }
}

fn shift_rows_up(arr: &mut ChunkedMatrix, row_idx: usize, limit: usize) {
    let m = arr.cols;

    for i in row_idx..limit-1 {
        for j in 0..m {
            let next_row_val = arr.get(i+1, j);
            arr.set(i, j, next_row_val);
        }
    }
}

fn argmax_abs_2d(arr: &ChunkedMatrix, rows: usize, cols: usize) -> (usize, usize) {
    let max_rows = min(rows, arr.rows); // Safety bounds
    let max_cols = min(cols, arr.cols); // Safety bounds

    let mut max_idx = (0, 0);
    let mut max_val = 0.0;

    for i in 0..max_rows {
        for j in 0..max_cols {
            let abs_val = arr.get(i, j).abs();
            if abs_val > max_val {
                max_val = abs_val;
                max_idx = (i, j);
            }
        }
    }

    max_idx
}

#[inline(always)]
fn argmax_abs_1d(arr: ArrayView1<f64>) -> usize {
    let mut max_idx = 0;
    let mut max_val = 0.0;

    for (i, &val) in arr.iter().enumerate() {
        let abs_val = val.abs();
        if abs_val > max_val {
            max_val = abs_val;
            max_idx = i;
        }
    }

    max_idx
}

fn compute_projected_column<M: MatrixAccess>(
    matrix: &M,
    col_idx: usize,
    q: &ChunkedMatrix,
    r: &ChunkedMatrix,
    rank: usize
) -> Array1<f64> {
    let n = min(matrix.rows(), BOUND);
    let mut result = Array1::<f64>::zeros(n);

    // Copy column from matrix
    for i in 0..n {
        result[i] = matrix.get(i, col_idx);
    }

    if rank > 0 {
        let safe_rank = min(rank, q.cols);

        for i in 0..n {
            let mut sum = 0.0;
            for k in 0..safe_rank {
                if k < r.rows && col_idx < r.cols {
                    sum += q.get(i, k) * r.get(k, col_idx);
                }
            }
            result[i] -= sum;
        }
    }

    result
}

fn compute_projected_row<M: MatrixAccess>(
    matrix: &M,
    row_idx: usize,
    q: &ChunkedMatrix,
    r: &ChunkedMatrix,
    rank: usize
) -> Array1<f64> {
    let m = min(matrix.cols(), BOUND);
    let mut result = Array1::<f64>::zeros(m);

    for j in 0..m {
        result[j] = matrix.get(row_idx, j);
    }

    if rank > 0 {
        let safe_rank = min(rank, q.cols);

        for j in 0..m {
            let mut sum = 0.0;
            for k in 0..safe_rank {
                if row_idx < q.rows && k < r.rows && j < r.cols {
                    sum += q.get(row_idx, k) * r.get(k, j);
                }
            }
            result[j] -= sum;
        }
    }

    result
}

#[inline(always)]
fn dot_product(a: &ArrayView1<f64>, b: &ArrayView1<f64>, n: usize) -> f64 {
    let safe_n = min(n, min(a.len(), b.len()));

    let mut sum = 0.0;
    for i in 0..safe_n {
        sum += a[i] * b[i];
    }
    sum
}

#[inline(always)]
fn squared_norm(a: &ArrayView1<f64>, n: usize) -> f64 {
    let safe_n = min(n, a.len());

    let mut sum = 0.0;
    for i in 0..safe_n {
        sum += a[i] * a[i];
    }
    sum
}

pub fn estimate_cross_approx_memory<M: MatrixAccess>(
    matrix: &M,
    r: Option<usize>
) -> usize {
    let n = matrix.rows();
    let m = matrix.cols();
    let max_rank = r.unwrap_or_else(|| min(n, m));

    let q_matrix_size = n * max_rank * size_of::<f64>();
    let r_matrix_size = max_rank * m * size_of::<f64>();

    let cols_size = n * max_rank * 2 * size_of::<f64>();
    let rows_size = max_rank * 2 * m * size_of::<f64>();

    let temp_arrays_size = (n + m) * 5 * size_of::<f64>();
    let overhead = 10 * 1024 * 1024; // 10MB overhead for various small allocations

    q_matrix_size + r_matrix_size + cols_size + rows_size + temp_arrays_size + overhead
}

fn remove_column_safe(arr: &mut ChunkedMatrix, col_idx: usize, free_col: usize) {
    if col_idx >= free_col || free_col <= 1 {
        return;
    }

    let max_cols = arr.cols;

    let safe_free_col = min(free_col, max_cols);

    if col_idx >= safe_free_col {
        return;
    }

    shift_columns_left(arr, col_idx, safe_free_col);
}

fn remove_row_safe(arr: &mut ChunkedMatrix, row_idx: usize, free_row: usize) {
    if row_idx >= free_row || free_row <= 1 {
        return;
    }

    let max_rows = arr.rows;

    let safe_free_row = min(free_row, max_rows);

    if row_idx >= safe_free_row {
        return;
    }

    shift_rows_up(arr, row_idx, safe_free_row);
}

fn update_matrices_safe(
    cols: &mut ChunkedMatrix,
    rows: &mut ChunkedMatrix,
    q_col: &ArrayView1<f64>,
    r_row: &ArrayView1<f64>,
    ind_cols: &[usize],
    ind_rows: &[usize],
    free_col: usize,
    free_row: usize,
) {
    let safe_free_col = min(free_col, ind_cols.len());
    let safe_free_row = min(free_row, ind_rows.len());

    let r_vals = safe_transform_indices(
        safe_free_col,
        ind_cols,
        r_row.len(),
        |idx| r_row[idx]
    );

    let q_vals = safe_transform_indices(
        safe_free_row,
        ind_rows,
        q_col.len(),
        |idx| q_col[idx]
    );

    for (j, r_val) in r_vals.iter().enumerate() {
        for i in 0..cols.rows {
            let current = cols.get(i, j);
            cols.set(i, j, current - q_col[i] * r_val);
        }
    }

    for (i, q_val) in q_vals.iter().enumerate() {
        for j in 0..rows.cols {
            let current = rows.get(i, j);
            rows.set(i, j, current - q_val * r_row[j]);
        }
    }
}

/// Memory-optimized cross approximation function
///
/// # Arguments
/// * `matrix` - The input matrix to approximate
/// * `r` - The rank of the approximation (default: min(n, m))
/// * `eps` - The error tolerance (default: 0.0)
///
/// # Returns
/// * `Q` - The left factor matrix as a ChunkedMatrix
/// * `R` - The right factor matrix as a ChunkedMatrix
/// * `rank` - The actual rank achieved
pub fn cross_approx<M: MatrixAccess>(
    matrix: &M,
    r: Option<usize>,
    eps: Option<f64>
) -> (ChunkedMatrix, ChunkedMatrix, usize) {
    let n = matrix.rows();
    let m = matrix.cols();

    if n >  BOUND || m > BOUND {
        panic!("Matrix dimensions too large: {}x{}", n, m);
    }

    if n == 0 || m == 0 {
        return (ChunkedMatrix::zeros(n, 0), ChunkedMatrix::zeros(0, m), 0);
    }

    let max_rank = r.unwrap_or_else(|| min(n, m));
    let eps = eps.unwrap_or(0.0);

    let max_rank = min(max_rank, min(n, m));
    let max_rank = min(max_rank, 1000);

    let max_rank = max(max_rank, 1);

    let mut q_matrix = ChunkedMatrix::zeros(n, max_rank);
    let mut r_matrix = ChunkedMatrix::zeros(max_rank, m);

    let initial_cols_capacity = min(32, max_rank);
    let initial_rows_capacity = initial_cols_capacity;

    let mut cols = ChunkedMatrix::zeros(n, initial_cols_capacity);
    let mut rows = ChunkedMatrix::zeros(initial_rows_capacity, m);

    let mut ind_cols = Vec::with_capacity(initial_cols_capacity);
    let mut ind_rows = Vec::with_capacity(initial_rows_capacity);

    let first_col = thread_rng().gen_range(0..m);
    ind_cols.push(first_col);

    for i in 0..n {
        cols.set(i, 0, matrix.get(i, first_col));
    }
    let mut free_col = 1;

    let mut temp_col = Array1::<f64>::zeros(n);
    let mut temp_row = Array1::<f64>::zeros(m);
    let mut temp_q_col = Array1::<f64>::zeros(n);
    let mut temp_r_row = Array1::<f64>::zeros(m);

    cols.slice_column_into(0, &mut temp_col);

    if temp_col.is_empty() {
        return (q_matrix, r_matrix, 0);
    }

    let first_row = argmax_abs_1d(temp_col.view());
    ind_rows.push(first_row);

    for j in 0..m {
        rows.set(0, j, matrix.get(first_row, j));
    }
    let mut free_row = 1;

    let mut approx_norm_squared = 0.0;
    let mut best_pivot: f64 = 0.0;

    for i in 0..max_rank {
        if free_col == 0 && free_row == 0 {
            return (q_matrix, r_matrix, i);
        }

        if i > 0 && i % 10 == 0 {
            cols.shrink_to_fit();
            rows.shrink_to_fit();
        }

        let (max_row_from_cols, max_col_from_cols) = if free_col > 0 {
            argmax_abs_2d(&cols, min(n, cols.rows), min(free_col, cols.cols))
        } else {
            (0, 0)
        };

        if free_row == 0 || rows.cols == 0 {
            return (q_matrix, r_matrix, i);
        }

        rows.slice_row_into(0, &mut temp_row);

        if temp_row.is_empty() {
            return (q_matrix, r_matrix, i);
        }

        let max_col_from_rows = argmax_abs_1d(temp_row.view());
        let max_row_from_rows = 0; // We're looking at the first row

        let pivot;

        let safe_row_index_option = if max_row_from_rows < ind_rows.len() {
            Some(ind_rows[max_row_from_rows])
        } else {
            None
        };

        let safe_col_index_option = if max_col_from_cols < ind_cols.len() {
            Some(ind_cols[max_col_from_cols])
        } else {
            None
        };

        if safe_row_index_option.is_none() || safe_col_index_option.is_none() {
            if free_col < initial_cols_capacity {
                let col_idx = thread_rng().gen_range(0..m);
                ind_cols.push(col_idx);

                temp_col = compute_projected_column(matrix, col_idx, &q_matrix, &r_matrix, i + 1);
                cols.set_column(free_col, &temp_col);
                free_col += 1;
            }

            if free_row < initial_rows_capacity {
                let row_idx = thread_rng().gen_range(0..n);
                ind_rows.push(row_idx);

                temp_row = compute_projected_row(matrix, row_idx, &q_matrix, &r_matrix, i + 1);
                rows.set_row(free_row, &temp_row);
                free_row += 1;
            }

            continue;
        }

        let safe_row_index = safe_row_index_option.unwrap();
        let safe_col_index = safe_col_index_option.unwrap();


        if max_row_from_cols == safe_row_index && max_col_from_rows == safe_col_index && free_col > 0 && free_row > 0 {
            if max_col_from_cols < free_col {
                pivot = cols.get(max_row_from_cols, max_col_from_cols);
            } else {
                continue;
            }

            //if pivot.abs() < 1e-12 { break; }

            for j in 0..n {
                q_matrix.set(j, i, cols.get(j, max_col_from_cols) / pivot);
            }
            for j in 0..m {
                r_matrix.set(i, j, rows.get(max_row_from_rows, j));
            }

            if max_col_from_cols < free_col {
                remove_column_safe(&mut cols, max_col_from_cols, free_col);
            }

            if max_row_from_rows < free_row {
                remove_row_safe(&mut rows, max_row_from_rows, free_row);
            }

            if max_col_from_cols < ind_cols.len() {
                ind_cols.remove(max_col_from_cols);
            }

            if max_row_from_rows < ind_rows.len() {
                ind_rows.remove(max_row_from_rows);
            }

            free_col = if free_col > 0 { free_col - 1 } else { 0 };
            free_row = if free_row > 0 { free_row - 1 } else { 0 };
        } else if free_col > 0 && max_col_from_cols < free_col &&
            (cols.get(max_row_from_cols, max_col_from_cols).abs() >
                rows.get(max_row_from_rows, max_col_from_rows).abs()) {

            temp_row = compute_projected_row(matrix, max_row_from_cols, &q_matrix, &r_matrix, i);

            if free_row < rows.rows {
                rows.set_row(free_row, &temp_row);
            } else {
                continue;
            }

            rows.slice_row_into(free_row, &mut temp_row);
            let col = argmax_abs_1d(temp_row.view());

            if col == safe_col_index && max_col_from_cols < free_col {
                pivot = cols.get(max_row_from_cols, max_col_from_cols);
                if pivot.abs() < 1e-12 { break; }

                for j in 0..n {
                    q_matrix.set(j, i, cols.get(j, max_col_from_cols) / pivot);
                }
                for j in 0..m {
                    r_matrix.set(i, j, rows.get(free_row, j));
                }

                if max_col_from_cols < free_col {
                    remove_column_safe(&mut cols, max_col_from_cols, free_col);
                }

                // Safely remove index with bounds checking
                if max_col_from_cols < ind_cols.len() {
                    ind_cols.remove(max_col_from_cols);
                }

                free_col = if free_col > 0 { free_col - 1 } else { 0 };
            } else {
                ind_rows.push(max_row_from_cols);
                free_row += 1;

                temp_col = compute_projected_column(matrix, col, &q_matrix, &r_matrix, i);

                if temp_col.is_empty() {
                    continue;
                }

                let row = argmax_abs_1d(temp_col.view());

                if row < temp_col.len() {
                    pivot = temp_col[row];
                } else {
                    continue;
                }

                //if pivot.abs() < 1e-12 { break; }

                for j in 0..n {
                    q_matrix.set(j, i, temp_col[j]);
                }

                temp_row = compute_projected_row(matrix, row, &q_matrix, &r_matrix, i);
                for j in 0..m {
                    r_matrix.set(i, j, temp_row[j] / pivot);
                }
            }
        } else if free_row > 0 && max_row_from_rows < free_row {
            if max_col_from_rows < m {
                temp_col = compute_projected_column(matrix, max_col_from_rows, &q_matrix, &r_matrix, i);
            } else {
                continue;
            }

            if free_col < cols.cols {
                cols.set_column(free_col, &temp_col);
            } else {
                continue;
            }

            cols.slice_column_into(free_col, &mut temp_col);

            if temp_col.is_empty() {
                continue;
            }

            let row = argmax_abs_1d(temp_col.view());

            if row == safe_row_index && max_row_from_rows < free_row {
                pivot = rows.get(max_row_from_rows, max_col_from_rows);
                if pivot.abs() < 1e-12 { break; }

                for j in 0..m {
                    r_matrix.set(i, j, rows.get(max_row_from_rows, j) / pivot);
                }
                for j in 0..n {
                    q_matrix.set(j, i, cols.get(j, free_col));
                }

                if max_row_from_rows < free_row {
                    remove_row_safe(&mut rows, max_row_from_rows, free_row);
                }

                if max_row_from_rows < ind_rows.len() {
                    ind_rows.remove(max_row_from_rows);
                }

                free_row = if free_row > 0 { free_row - 1 } else { 0 };
            } else {
                ind_cols.push(max_col_from_rows);
                free_col += 1;

                temp_row = compute_projected_row(matrix, row, &q_matrix, &r_matrix, i);
                for j in 0..m {
                    r_matrix.set(i, j, temp_row[j]);
                }

                let col = argmax_abs_1d(temp_row.view());

                if i < r_matrix.rows && col < r_matrix.cols {
                    pivot = r_matrix.get(i, col);
                } else {
                    pivot = 0.0;
                }

                if pivot.abs() < 1e-12 { break; }

                if col < m {
                    temp_col = compute_projected_column(matrix, col, &q_matrix, &r_matrix, i);
                } else {
                    continue;
                }

                for j in 0..n {
                    q_matrix.set(j, i, temp_col[j] / pivot);
                }
            }
        } else {
            if free_col < initial_cols_capacity && m > 0 {
                let col_idx = thread_rng().gen_range(0..m);
                ind_cols.push(col_idx);

                temp_col = compute_projected_column(matrix, col_idx, &q_matrix, &r_matrix, i + 1);
                if free_col < cols.cols {
                    cols.set_column(free_col, &temp_col);
                    free_col += 1;
                }
            }

            if free_row < initial_rows_capacity && n > 0 {
                let row_idx = thread_rng().gen_range(0..n);
                ind_rows.push(row_idx);

                temp_row = compute_projected_row(matrix, row_idx, &q_matrix, &r_matrix, i + 1);
                if free_row < rows.rows {
                    rows.set_row(free_row, &temp_row);
                    free_row += 1;
                }
            }

            continue;
        }

        best_pivot = best_pivot.max(pivot.abs());

        q_matrix.slice_column_into(i, &mut temp_q_col);
        r_matrix.slice_row_into(i, &mut temp_r_row);

        let safe_ind_cols = &ind_cols;
        let safe_ind_rows = &ind_rows;

        update_matrices_safe(
            &mut cols,
            &mut rows,
            &temp_q_col.view(),
            &temp_r_row.view(),
            safe_ind_cols,
            safe_ind_rows,
            free_col,
            free_row
        );

        if free_col == 0 && m > 0 {
            let col_idx = thread_rng().gen_range(0..m);
            ind_cols.push(col_idx);

            temp_col = compute_projected_column(matrix, col_idx, &q_matrix, &r_matrix, i + 1);
            if cols.cols > 0 {
                cols.set_column(0, &temp_col);
                free_col = 1;
            }
        }

        if free_row == 0 && n > 0 {
            let row_idx = thread_rng().gen_range(0..n);
            ind_rows.push(row_idx);

            temp_row = compute_projected_row(matrix, row_idx, &q_matrix, &r_matrix, i + 1);
            if rows.rows > 0 {
                rows.set_row(0, &temp_row);
                free_row = 1;
            }
        }

        q_matrix.slice_column_into(i, &mut temp_q_col);
        r_matrix.slice_row_into(i, &mut temp_r_row);

        let q_norm = squared_norm(&temp_q_col.view(), n);
        let r_norm = squared_norm(&temp_r_row.view(), m);

        for j in 0..i {
            q_matrix.slice_column_into(j, &mut temp_q_col);
            r_matrix.slice_row_into(j, &mut temp_r_row);

            let q_dot = dot_product(&temp_q_col.view(), &q_matrix.column(i).view(), n);
            let r_dot = dot_product(&temp_r_row.view(), &r_matrix.row(i).view(), m);

            approx_norm_squared += 2.0 * q_dot * r_dot;
        }

        approx_norm_squared += q_norm * r_norm;

        if eps > 0.0 {
            let error_bound = pivot.abs() * (((m - i - free_row) * (n - i - free_col)) as f64).sqrt();
            if approx_norm_squared.sqrt() * eps >= error_bound {
                return (q_matrix, r_matrix, i + 1);
            }
        }
    }
    q_matrix.shrink_to_fit();
    r_matrix.shrink_to_fit();

    (q_matrix, r_matrix, max_rank)
}