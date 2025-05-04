use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array2, ArrayView2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::{Uniform, Normal};
use cross::cross_optimized::{cross_approx, MatrixAccess, estimate_cross_approx_memory};
use std::time::{Duration, Instant};
use std::mem::{size_of, size_of_val};
use std::cmp::min;

// Memory tracking code
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

#[global_allocator]
static ALLOCATOR: MemoryTracker = MemoryTracker;

struct MemoryTracker;

static CURRENT_MEMORY: AtomicUsize = AtomicUsize::new(0);
static PEAK_MEMORY: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for MemoryTracker {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);

        if !ptr.is_null() {
            let size = layout.size();
            let current = CURRENT_MEMORY.fetch_add(size, Ordering::SeqCst) + size;

            let mut peak = PEAK_MEMORY.load(Ordering::SeqCst);
            while current > peak {
                match PEAK_MEMORY.compare_exchange(peak, current, Ordering::SeqCst, Ordering::SeqCst) {
                    Ok(_) => break,
                    Err(p) => peak = p,
                }
            }
        }

        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if !ptr.is_null() {
            CURRENT_MEMORY.fetch_sub(layout.size(), Ordering::SeqCst);
        }

        System.dealloc(ptr, layout);
    }
}

fn reset_memory_tracking() {
    CURRENT_MEMORY.store(0, Ordering::SeqCst);
    PEAK_MEMORY.store(0, Ordering::SeqCst);
}

fn get_peak_memory() -> usize {
    PEAK_MEMORY.load(Ordering::SeqCst)
}

fn get_current_memory() -> usize {
    CURRENT_MEMORY.load(Ordering::SeqCst)
}

fn format_memory_size(size: usize) -> String {
    if size < 1024 {
        format!("{} B", size)
    } else if size < 1024 * 1024 {
        format!("{:.2} KB", size as f64 / 1024.0)
    } else if size < 1024 * 1024 * 1024 {
        format!("{:.2} MB", size as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", size as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

// Matrix implementations for benchmarking
struct DenseMatrix<'a> {
    data: ArrayView2<'a, f64>,
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

// Матрица Гильберта - вычисляется "на лету" без хранения в памяти
struct HilbertMatrix {
    rows: usize,
    cols: usize,
}

impl MatrixAccess for HilbertMatrix {
    #[inline(always)]
    fn get(&self, row: usize, col: usize) -> f64 {
        1.0 / ((row + col + 1) as f64)
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

// Матрица низкого ранга - эффективная реализация для больших размеров
struct LowRankMatrix {
    left: Array2<f64>,
    right: Array2<f64>,
    rows: usize,
    cols: usize,
}

impl LowRankMatrix {
    fn new(n: usize, m: usize, rank: usize) -> Self {
        // Создаем две случайные матрицы размера n x rank и rank x m
        let left = Array2::<f64>::random((n, rank), Normal::new(0.0, 1.0).unwrap());
        let right = Array2::<f64>::random((rank, m), Normal::new(0.0, 1.0).unwrap());

        Self {
            left,
            right,
            rows: n,
            cols: m,
        }
    }
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

// Разреженная матрица с экспоненциальным убыванием
struct ExponentialDecayMatrix {
    rows: usize,
    cols: usize,
    decay_rate: f64,
}

impl ExponentialDecayMatrix {
    fn new(n: usize, m: usize, decay_rate: f64) -> Self {
        Self {
            rows: n,
            cols: m,
            decay_rate,
        }
    }
}

impl MatrixAccess for ExponentialDecayMatrix {
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

struct BenchResult {
    elapsed: Duration,
    peak_memory: usize,
    achieved_rank: usize,
}

fn bench_with_memory<F, R>(f: F) -> BenchResult
where
    F: FnOnce() -> (R, usize),
{
    reset_memory_tracking();
    let start = Instant::now();
    let (_, achieved_rank) = f();
    let elapsed = start.elapsed();
    let peak_memory = get_peak_memory();

    BenchResult {
        elapsed,
        peak_memory,
        achieved_rank,
    }
}

// Ручной бенчмарк для больших матриц с автоматическим выбором ранга (без использования Criterion)
fn manual_bench_large_matrices() {
    // Задаем размеры матриц
    let sizes = [(2048, 2048), (4096, 4096), (8192, 8192)];

    // Значения точности для автоматического определения ранга
    let epsilon_values = [1e-4, 1e-6, 1e-8];

    println!("\nLarge Matrix Benchmark with Auto Rank Selection (r = None):");
    println!("{:<15} {:<10} {:<15} {:<15} {:<15} {:<15}",
             "Matrix Size", "Epsilon", "Achieved Rank", "Time", "Peak Memory", "Matrix Type");
    println!("{}", "-".repeat(90));

    for &(n, m) in &sizes {
        // Для Hilbert и Exponential матриц выполняем тесты для всех размеров
        for &eps in &epsilon_values {
            // Создаем матрицу Гильберта
            let hilbert_matrix = HilbertMatrix { rows: n, cols: m };

            // Оцениваем необходимую память (с консервативной оценкой ранга)
            let estimated_rank = min(n, m) / 10; // Примерная оценка ранга
            let estimated_memory = estimate_cross_approx_memory(&hilbert_matrix, Some(estimated_rank));

            // Пропускаем, если требуется слишком много памяти
            let memory_limit = 6_000_000_000; // 6 ГБ
            if estimated_memory > memory_limit {
                println!(
                    "{:<15} {:<10} {:<15} {:<15} {:<15} {:<15}",
                    format!("{}x{}", n, m),
                    format!("{:.1e}", eps),
                    "N/A",
                    "SKIPPED",
                    format_memory_size(estimated_memory),
                    "Hilbert"
                );
                continue;
            }

            // Выполняем тест с автоматическим выбором ранга
            let result = bench_with_memory(|| {
                let (_, _, achieved_rank) = black_box(cross_approx(&hilbert_matrix, None, Some(eps)));
                ((), achieved_rank)
            });

            println!(
                "{:<15} {:<10} {:<15} {:<15} {:<15} {:<15}",
                format!("{}x{}", n, m),
                format!("{:.1e}", eps),
                result.achieved_rank,
                format!("{:.2?}", result.elapsed),
                format_memory_size(result.peak_memory),
                "Hilbert"
            );

            // Очистка памяти между тестами
            reset_memory_tracking();
            std::thread::sleep(Duration::from_secs(1));

            // Создаем матрицу с экспоненциальным убыванием
            let exp_matrix = ExponentialDecayMatrix::new(n, m, 0.1);

            // Тест для матрицы с экспоненциальным убыванием
            let result = bench_with_memory(|| {
                let (_, _, achieved_rank) = black_box(cross_approx(&exp_matrix, None, Some(eps)));
                ((), achieved_rank)
            });

            println!(
                "{:<15} {:<10} {:<15} {:<15} {:<15} {:<15}",
                format!("{}x{}", n, m),
                format!("{:.1e}", eps),
                result.achieved_rank,
                format!("{:.2?}", result.elapsed),
                format_memory_size(result.peak_memory),
                "Exponential"
            );

            // Очистка памяти между тестами
            reset_memory_tracking();
            std::thread::sleep(Duration::from_secs(1));
        }

        // Для матриц низкого ранга - только для размеров до 4096x4096
        if n <= 4096 {
            // Выбираем подходящие ранги для матрицы низкого ранга
            let low_ranks = [10, 20, 50];

            for &true_rank in &low_ranks {
                // Создаем матрицу низкого ранга
                let low_rank_matrix = LowRankMatrix::new(n, m, true_rank);

                // Тестируем с eps = 1e-6 для автоматического определения ранга
                let eps = 1e-6;

                let result = bench_with_memory(|| {
                    let (_, _, achieved_rank) = black_box(cross_approx(&low_rank_matrix, None, Some(eps)));
                    ((), achieved_rank)
                });

                println!(
                    "{:<15} {:<10} {:<15} {:<15} {:<15} {:<15}",
                    format!("{}x{}", n, m),
                    format!("{:.1e}", eps),
                    result.achieved_rank,
                    format!("{:.2?}", result.elapsed),
                    format_memory_size(result.peak_memory),
                    format!("Low Rank ({})", true_rank)
                );

                // Очистка памяти между тестами
                reset_memory_tracking();
                std::thread::sleep(Duration::from_secs(1));
            }
        }
    }

    println!("{}", "-".repeat(90));
}

// Тест для малых матриц с Criterion
fn bench_small_matrices(c: &mut Criterion) {
    let mut group = c.benchmark_group("Small Matrices");

    // Необходимые настройки для Criterion
    group.sample_size(10); // Минимальное количество измерений
    group.measurement_time(Duration::from_secs(5));

    // Используем только малые матрицы для Criterion
    let sizes = [(2048, 2048), (4096, 4096), (8192, 8192)];
    let epsilon_values = [1e-6];

    for &(n, m) in &sizes {
        for &eps in &epsilon_values {
            // Создаем матрицу Гильберта для тестирования
            let hilbert_matrix = HilbertMatrix { rows: n, cols: m };

            // Добавляем бенчмарк в Criterion
            let id = BenchmarkId::new(format!("Hilbert {}x{}", n, m), format!("{:.1e}", eps));
            group.bench_with_input(id, &eps, |b, &eps| {
                b.iter(|| {
                    black_box(cross_approx(&hilbert_matrix, None, Some(eps)))
                });
            });

            // Экспоненциальная матрица
            let exp_matrix = ExponentialDecayMatrix::new(n, m, 0.1);

            let id = BenchmarkId::new(format!("Exponential {}x{}", n, m), format!("{:.1e}", eps));
            group.bench_with_input(id, &eps, |b, &eps| {
                b.iter(|| {
                    black_box(cross_approx(&exp_matrix, None, Some(eps)))
                });
            });
        }
    }

    group.finish();
}

// Тестирование скорости доступа к разным типам матриц
fn bench_matrix_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix Access Methods");

    // Необходимая настройка для Criterion
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));

    // Для теста доступа используем матрицу небольшого размера
    let n = 512;
    let m = 512;

    // Создаем разные типы матриц
    let hilbert_matrix = HilbertMatrix { rows: n, cols: m };
    let exp_matrix = ExponentialDecayMatrix::new(n, m, 0.1);
    let low_rank_matrix = LowRankMatrix::new(n, m, 20);

    // Вычисляем размер в памяти
    let hilbert_size = size_of_val(&hilbert_matrix);
    let exp_size = size_of_val(&exp_matrix);
    let low_rank_size = size_of_val(&low_rank_matrix) +
        low_rank_matrix.left.len() * size_of::<f64>() +
        low_rank_matrix.right.len() * size_of::<f64>();

    println!("\nMatrix Memory Usage:");
    println!("{:<20} {:<15}", "Matrix Type", "Memory Size");
    println!("{}", "-".repeat(40));
    println!("{:<20} {:<15}", "Hilbert Matrix", format_memory_size(hilbert_size));
    println!("{:<20} {:<15}", "Exponential Matrix", format_memory_size(exp_size));
    println!("{:<20} {:<15}", "Low Rank Matrix", format_memory_size(low_rank_size));
    println!("{}", "-".repeat(40));

    // Тест скорости доступа (100x100 подматрица)
    group.bench_function("Hilbert Matrix", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for i in 0..100 {
                for j in 0..100 {
                    sum += black_box(hilbert_matrix.get(i, j));
                }
            }
            sum
        })
    });

    group.bench_function("Exponential Matrix", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for i in 0..100 {
                for j in 0..100 {
                    sum += black_box(exp_matrix.get(i, j));
                }
            }
            sum
        })
    });

    group.bench_function("Low Rank Matrix", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for i in 0..100 {
                for j in 0..100 {
                    sum += black_box(low_rank_matrix.get(i, j));
                }
            }
            sum
        })
    });

    group.finish();
}

// Функция для запуска всех бенчмарков
fn run_all_benchmarks() {
    // Запускаем ручной бенчмарк для больших матриц
    manual_bench_large_matrices();
}

criterion_group!(
    benches,
    bench_small_matrices,
    bench_matrix_access
);
criterion_main!(benches);