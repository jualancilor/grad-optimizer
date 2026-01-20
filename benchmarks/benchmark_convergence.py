"""
Benchmark konvergensi untuk gradient descent optimizer.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.function import Quadratic, Rosenbrock, Rastrigin
from src.optimizer import gradient_descent
from src.gradient import gradient_check


def benchmark_function(func, x_init, learning_rate, max_iter=5000, name=""):
    """
    Benchmark konvergensi pada satu fungsi.
    """
    print(f"\n{'='*60}")
    print(f"Benchmark: {name or func.name}")
    print(f"{'='*60}")
    print(f"Dimensi: {func.dim}")
    print(f"Titik awal: {x_init}")
    print(f"Learning rate: {learning_rate}")
    print(f"Max iterasi: {max_iter}")

    # Gradient check dulu
    is_ok, rel_err, _ = gradient_check(func, func.gradient, x_init)
    print(f"\nGradient check: {'PASS' if is_ok else 'FAIL'} (error: {rel_err:.2e})")

    # Jalankan dengan analytical gradient
    print("\n--- Analytical Gradient ---")
    x_opt_ana, hist_ana = gradient_descent(
        f=func,
        x_init=x_init,
        learning_rate=learning_rate,
        max_iter=max_iter,
        grad_func=func.gradient
    )
    iter_ana = len(hist_ana['f_values']) - 1
    f_final_ana = hist_ana['f_values'][-1]

    print(f"Iterasi: {iter_ana}")
    print(f"x_optimal: {[round(xi, 6) for xi in x_opt_ana]}")
    print(f"f(x_optimal): {f_final_ana:.6e}")

    # Jalankan dengan numerical gradient
    print("\n--- Numerical Gradient ---")
    x_opt_num, hist_num = gradient_descent(
        f=func,
        x_init=x_init,
        learning_rate=learning_rate,
        max_iter=max_iter,
        use_numerical=True
    )
    iter_num = len(hist_num['f_values']) - 1
    f_final_num = hist_num['f_values'][-1]

    print(f"Iterasi: {iter_num}")
    print(f"x_optimal: {[round(xi, 6) for xi in x_opt_num]}")
    print(f"f(x_optimal): {f_final_num:.6e}")

    # Bandingkan dengan global minimum jika ada
    if func.global_minimum is not None:
        print(f"\n--- Perbandingan dengan Global Minimum ---")
        print(f"Global minimum: {func.global_minimum}")
        print(f"f_min teoritis: {func.f_min}")

        error_ana = sum((x_opt_ana[i] - func.global_minimum[i])**2
                        for i in range(func.dim)) ** 0.5
        error_num = sum((x_opt_num[i] - func.global_minimum[i])**2
                        for i in range(func.dim)) ** 0.5
        print(f"Error (analytical): {error_ana:.6e}")
        print(f"Error (numerical): {error_num:.6e}")

    return {
        'name': name or func.name,
        'analytical': {'iterations': iter_ana, 'f_final': f_final_ana, 'x_opt': x_opt_ana},
        'numerical': {'iterations': iter_num, 'f_final': f_final_num, 'x_opt': x_opt_num}
    }


def run_all_benchmarks():
    """
    Jalankan semua benchmark.
    """
    print("=" * 60)
    print("BENCHMARK KONVERGENSI GRADIENT DESCENT")
    print("=" * 60)

    results = []

    # 1. Quadratic (convex, mudah)
    A = [[2.0, 0.0], [0.0, 2.0]]
    b = [1.0, 1.0]
    quad = Quadratic(A, b)
    results.append(benchmark_function(
        quad,
        x_init=[5.0, 5.0],
        learning_rate=0.1,
        name="Quadratic (convex)"
    ))

    # 2. Rosenbrock 2D (non-convex, classic)
    rosen = Rosenbrock(dim=2)
    results.append(benchmark_function(
        rosen,
        x_init=[-1.0, 1.0],
        learning_rate=0.001,
        max_iter=10000
    ))

    # 3. Rastrigin 2D (highly multimodal)
    rast = Rastrigin(dim=2)
    results.append(benchmark_function(
        rast,
        x_init=[0.5, 0.5],
        learning_rate=0.01,
        max_iter=1000
    ))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Function':<25} {'Analytical':<15} {'Numerical':<15}")
    print(f"{'':25} {'(iters/f_val)':<15} {'(iters/f_val)':<15}")
    print("-" * 60)
    for r in results:
        ana = f"{r['analytical']['iterations']}/{r['analytical']['f_final']:.2e}"
        num = f"{r['numerical']['iterations']}/{r['numerical']['f_final']:.2e}"
        print(f"{r['name']:<25} {ana:<15} {num:<15}")

    return results


if __name__ == "__main__":
    run_all_benchmarks()
