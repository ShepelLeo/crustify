use std::time::Duration;
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use cross::cross::cross_approx;
use ndarray::Array2;
use rand::Rng;

fn random_matrix(n: usize, m: usize) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    let data: Vec<f64> = (0..n * m).map(|_| rng.gen_range(-1.0..1.0)).collect();
    Array2::from_shape_vec((n, m), data).unwrap()
}

fn bench_cross(c: &mut Criterion) {
    let n = 4096;
    let m = 4096;
    let mat = random_matrix(n, m);

    let mut group = c.benchmark_group("CrossApproximation4096");
    group.sample_size(30);
    group.measurement_time(Duration::from_secs(20));
    for r in [50, 100, 250, 500, 750, 1000] {
        group.bench_with_input(BenchmarkId::from_parameter(r), &mat, |b, mat| {
            b.iter(|| {
                let _ = cross_approx(mat.into(), Some(r), Some(1e-6));
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_cross);
criterion_main!(benches);
