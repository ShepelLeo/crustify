use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use ndarray_rand::rand_distr::Uniform;
use rand::{thread_rng, Rng};
use std::cmp::min;
use std::ops::Range;

fn argmax_abs_2d(arr: ArrayView2<f64>) -> (usize, usize) {
    arr.indexed_iter()
        .max_by(|(_, &a), (_, &b)| a.abs().partial_cmp(&b.abs()).unwrap())
        .map(|((i, j), _)| (i, j))
        .unwrap_or((0, 0))
}

fn argmax_abs_1d(arr: ArrayView1<f64>) -> usize {
    arr.iter()
        .enumerate()
        .max_by(|(_, &a), (_, &b)| a.abs().partial_cmp(&b.abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn find_index(arr: &[usize], value: usize) -> Option<usize> {
    arr.iter().position(|&x| x == value)
}

fn update_matrices(
    cols: &mut Array2<f64>,
    rows: &mut Array2<f64>,
    q_col: ArrayView1<f64>,
    r_row: ArrayView1<f64>,
    ind_cols: &[usize],
    ind_rows: &[usize],
    free_col: usize,
    free_row: usize,
) {
    let r_vals: Vec<f64> = ind_cols[..free_col]
        .iter()
        .map(|&col_idx| r_row[col_idx])
        .collect();

    let q_vals: Vec<f64> = ind_rows[..free_row]
        .iter()
        .map(|&row_idx| q_col[row_idx])
        .collect();

    (0..free_col).for_each(|j| {
        let r_val = r_vals[j];
        cols.column_mut(j)
            .iter_mut()
            .zip(q_col.iter())
            .for_each(|(col_val, &q_val)| {
                *col_val -= q_val * r_val;
            });
    });

    (0..free_row).for_each(|i| {
        let q_val = q_vals[i];
        rows.row_mut(i)
            .iter_mut()
            .zip(r_row.iter())
            .for_each(|(row_val, &r_val)| {
                *row_val -= q_val * r_val;
            });
    });
}

fn compute_projected_column(
    matrix: ArrayView2<f64>,
    col_idx: usize,
    q: ArrayView2<f64>,
    r: ArrayView2<f64>,
    rank: usize
) -> Array1<f64> {
    let n = matrix.shape()[0];
    let mut result = Array1::<f64>::zeros(n);

    result.assign(&matrix.column(col_idx));

    if rank > 0 {
        let q_slice = q.slice(s![.., ..rank]);
        let r_col = r.slice(s![..rank, col_idx]);

        (0..n).for_each(|i| {
            let sum = (0..rank)
                .map(|k| q_slice[[i, k]] * r_col[k])
                .sum::<f64>();
            result[i] -= sum;
        });
    }

    result
}

fn compute_projected_row(
    matrix: ArrayView2<f64>,
    row_idx: usize,
    q: ArrayView2<f64>,
    r: ArrayView2<f64>,
    rank: usize
) -> Array1<f64> {
    let m = matrix.shape()[1];
    let mut result = Array1::<f64>::zeros(m);

    result.assign(&matrix.row(row_idx));

    if rank > 0 {
        let q_row = q.slice(s![row_idx, ..rank]);
        let r_slice = r.slice(s![..rank, ..]);

        (0..m).for_each(|j| {
            let sum = (0..rank)
                .map(|k| q_row[k] * r_slice[[k, j]])
                .sum::<f64>();
            result[j] -= sum;
        });
    }

    result
}


/// # Arguments
/// * `matrix` - The input matrix to approximate
/// * `r` - The rank of the approximation (default: min(n, m))
/// * `eps` - The error tolerance (default: 0.0)
///
/// # Returns
/// * `Q` - The left factor matrix
/// * `R` - The right factor matrix
/// * `rank` - The actual rank achieved
pub fn cross_approx(
    matrix: ArrayView2<f64>,
    r: Option<usize>,
    eps: Option<f64>
) -> (Array2<f64>, Array2<f64>, usize) {
    let n = matrix.shape()[0];
    let m = matrix.shape()[1];

    let r = r.unwrap_or_else(|| min(n, m));
    let eps = eps.unwrap_or(0.0);

    let mut q_matrix = Array2::<f64>::zeros((n, r));
    let mut r_matrix = Array2::<f64>::zeros((r, m));

    let mut i = 0;
    let mut err = eps + 1.0;

    let mut cols = Array2::<f64>::zeros((n, r + 1));
    let mut ind_cols = vec![m; r + 1];
    ind_cols[0] = thread_rng().sample(Uniform::new(0, m));
    cols.column_mut(0).assign(&matrix.column(ind_cols[0]));
    let mut free_col = 1;

    let mut vec_cols: Vec<usize> = (0..m).collect();

    let mut rows = Array2::<f64>::zeros((r + 1, m));
    let mut ind_rows = vec![n; r + 1];
    ind_rows[0] = argmax_abs_1d(cols.column(0));
    rows.row_mut(0).assign(&matrix.row(ind_rows[0]));
    let mut free_row = 1;

    let mut vec_rows: Vec<usize> = (0..n).collect();

    while i < r && err > eps {
        let (max_row_from_cols, max_col_from_cols) = argmax_abs_2d(cols.slice(s![.., ..free_col]));
        let (max_row_from_rows, max_col_from_rows) = argmax_abs_2d(rows.slice(s![..free_row, ..]));

        if max_row_from_cols == ind_rows[max_row_from_rows] && max_col_from_rows == ind_cols[max_col_from_cols] {
            let pivot = cols[[max_row_from_cols, max_col_from_cols]];

            q_matrix.column_mut(i).assign(&(&cols.column(max_col_from_cols) / pivot));

            r_matrix.row_mut(i).assign(&rows.row(max_row_from_rows));

            remove_column_inplace(&mut cols, max_col_from_cols, free_col);
            remove_row_inplace(&mut rows, max_row_from_rows, free_row);

            ind_cols.remove(max_col_from_cols);
            ind_rows.remove(max_row_from_rows);

            free_col -= 1;
            free_row -= 1;

            update_matrices(
                &mut cols,
                &mut rows,
                q_matrix.column(i),
                r_matrix.row(i),
                &ind_cols,
                &ind_rows,
                free_col,
                free_row
            );

            if let Some(idx) = find_index(&vec_cols, max_col_from_rows) {
                vec_cols.remove(idx);
            }

            if let Some(idx) = find_index(&vec_rows, max_row_from_cols) {
                vec_rows.remove(idx);
            }
        } else if cols[[max_row_from_cols, max_col_from_cols]].abs() > rows[[max_row_from_rows, max_col_from_rows]].abs() {
            let temp_row = compute_projected_row(matrix, max_row_from_cols, q_matrix.view(), r_matrix.view(), i);
            rows.row_mut(free_row).assign(&temp_row);

            let col = argmax_abs_1d(rows.row(free_row));

            if col == ind_cols[max_col_from_cols] {
                let pivot = cols[[max_row_from_cols, max_col_from_cols]];
                q_matrix.column_mut(i).assign(&(&cols.column(max_col_from_cols) / pivot));
                r_matrix.row_mut(i).assign(&rows.row(free_row));
                remove_column_inplace(&mut cols, max_col_from_cols, free_col);
                ind_cols.remove(max_col_from_cols);
                free_col -= 1;
                if let Some(idx) = find_index(&vec_rows, max_row_from_cols) {
                    vec_rows.remove(idx);
                }
            } else {
                ind_rows[free_row] = max_row_from_cols;
                free_row += 1;

                let temp_col = compute_projected_column(matrix, col, q_matrix.view(), r_matrix.view(), i);
                let row = argmax_abs_1d(temp_col.view());
                let pivot = temp_col[row];

                q_matrix.column_mut(i).assign(&temp_col);

                let temp_row = compute_projected_row(matrix, row, q_matrix.view(), r_matrix.view(), i);
                r_matrix.row_mut(i).assign(&(&temp_row / pivot));

                if let Some(idx) = find_index(&vec_rows, row) {
                    vec_rows.remove(idx);
                }
            }

            update_matrices(
                &mut cols,
                &mut rows,
                q_matrix.column(i),
                r_matrix.row(i),
                &ind_cols,
                &ind_rows,
                free_col,
                free_row
            );

            if let Some(idx) = find_index(&vec_cols, col) {
                vec_cols.remove(idx);
            }
        } else {
            let temp_col = compute_projected_column(matrix, max_col_from_rows, q_matrix.view(), r_matrix.view(), i);
            cols.column_mut(free_col).assign(&temp_col);

            let row = argmax_abs_1d(cols.column(free_col));

            if row == ind_rows[max_row_from_rows] {
                let pivot = rows[[max_row_from_rows, max_col_from_rows]];

                r_matrix.row_mut(i).assign(&(&rows.row(max_row_from_rows) / pivot));

                q_matrix.column_mut(i).assign(&cols.column(free_col));

                remove_row_inplace(&mut rows, max_row_from_rows, free_row);
                ind_rows.remove(max_row_from_rows);
                free_row -= 1;

                if let Some(idx) = find_index(&vec_cols, max_col_from_rows) {
                    vec_cols.remove(idx);
                }
            } else {
                ind_cols[free_col] = max_col_from_rows;
                free_col += 1;

                let temp_row = compute_projected_row(matrix, row, q_matrix.view(), r_matrix.view(), i);
                r_matrix.row_mut(i).assign(&temp_row);

                let col = argmax_abs_1d(r_matrix.row(i));
                let pivot = r_matrix[[i, col]];

                let temp_col = compute_projected_column(matrix, col, q_matrix.view(), r_matrix.view(), i);

                for j in 0..n {
                    q_matrix[[j, i]] = temp_col[j] / pivot;
                }

                if let Some(idx) = find_index(&vec_cols, col) {
                    vec_cols.remove(idx);
                }
            }

            update_matrices(
                &mut cols,
                &mut rows,
                q_matrix.column(i),
                r_matrix.row(i),
                &ind_cols,
                &ind_rows,
                free_col,
                free_row
            );

            if let Some(idx) = find_index(&vec_rows, row) {
                vec_rows.remove(idx);
            }
        }

        i += 1;

        if free_col == 0 {
            let rand_idx = thread_rng().sample(Uniform::new(0, m - i));
            ind_cols[0] = vec_cols[rand_idx];

            let temp_col = compute_projected_column(matrix, ind_cols[0], q_matrix.view(), r_matrix.view(), i);
            cols.column_mut(0).assign(&temp_col);
            free_col = 1;
        }

        if free_row == 0 {
            let rand_idx = thread_rng().sample(Uniform::new(0, n - i));
            ind_rows[0] = vec_rows[rand_idx];

            let temp_row = compute_projected_row(matrix, ind_rows[0], q_matrix.view(), r_matrix.view(), i);
            rows.row_mut(0).assign(&temp_row);
            free_row = 1;
        }

        let norm_rows = calculate_norm(&rows, 0..free_row, 0..m, None);
        let norm_cols = calculate_norm(&cols, 0..n, 0..free_col, None);
        let norm_intersection = calculate_norm(&rows, 0..free_row, 0..free_col, Some(&ind_cols));

        let denominator = (n * free_col + m * free_row - free_col * free_row) as f64;
        err = ((norm_rows + norm_cols - norm_intersection) / denominator * (n - i) as f64 * (m - i) as f64).sqrt();
    }

    (q_matrix, r_matrix, i)
}

fn calculate_norm(matrix: &Array2<f64>, row_range: Range<usize>, col_range: Range<usize>, col_indices: Option<&[usize]>) -> f64 {
    row_range.flat_map(move |i| {
        (col_range.start..col_range.end).map(move |j| {
            let col_idx = col_indices.map_or(j, |indices| indices[j]);
            matrix[[i, col_idx]].powi(2)
        })
    }).sum()
}

fn remove_column_inplace(arr: &mut Array2<f64>, col_idx: usize, free_col: usize) {
    let n = arr.shape()[0];

    (col_idx..free_col-1).for_each(|j| {
        (0..n).for_each(|i| {
            arr[[i, j]] = arr[[i, j+1]];
        });
    });
}

fn remove_row_inplace(arr: &mut Array2<f64>, row_idx: usize, free_row: usize) {
    let m = arr.shape()[1];

    (row_idx..free_row-1).for_each(|i| {
        (0..m).for_each(|j| {
            arr[[i, j]] = arr[[i+1, j]];
        });
    });
}