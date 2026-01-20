"""
Benchmark konvergensi untuk gradient descent optimizer.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.function import Quadratic, Rosenbrock, Rastrigin
from src.optimizer import gradient_descent, momentum_gradient_descent
from src.gradient import gradient_check


def benchmark_function(func, x_init, learning_rate, momentum=0.9, max_iter=5000, name=""):
    """
    Benchmark konvergensi pada satu fungsi.
    """
    print(f"\n{'='*60}")
    print(f"Benchmark: {name or func.name}")
    print(f"{'='*60}")
    print(f"Dimensi: {func.dim}")
    print(f"Titik awal: {x_init}")
    print(f"Learning rate: {learning_rate}")
    print(f"Momentum: {momentum}")
    print(f"Max iterasi: {max_iter}")

    # Gradient check dulu
    is_ok, rel_err, _ = gradient_check(func, func.gradient, x_init)
    print(f"\nGradient check: {'PASS' if is_ok else 'FAIL'} (error: {rel_err:.2e})")

    # 1. Vanilla GD dengan analytical gradient
    print("\n--- Vanilla GD (Analytical) ---")
    x_opt_vanilla, hist_vanilla = gradient_descent(
        f=func,
        x_init=x_init,
        learning_rate=learning_rate,
        max_iter=max_iter,
        grad_func=func.gradient
    )
    iter_vanilla = len(hist_vanilla['f_values']) - 1
    f_final_vanilla = hist_vanilla['f_values'][-1]

    print(f"Iterasi: {iter_vanilla}")
    print(f"x_optimal: {[round(xi, 6) for xi in x_opt_vanilla]}")
    print(f"f(x_optimal): {f_final_vanilla:.6e}")

    # 2. Momentum GD dengan analytical gradient
    print("\n--- Momentum GD (Analytical) ---")
    x_opt_mom, hist_mom = momentum_gradient_descent(
        f=func,
        x_init=x_init,
        learning_rate=learning_rate,
        momentum=momentum,
        max_iter=max_iter,
        grad_func=func.gradient
    )
    iter_mom = len(hist_mom['f_values']) - 1
    f_final_mom = hist_mom['f_values'][-1]

    print(f"Iterasi: {iter_mom}")
    print(f"x_optimal: {[round(xi, 6) for xi in x_opt_mom]}")
    print(f"f(x_optimal): {f_final_mom:.6e}")

    # 3. Vanilla GD dengan numerical gradient
    print("\n--- Vanilla GD (Numerical) ---")
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

    # 4. Momentum GD dengan numerical gradient
    print("\n--- Momentum GD (Numerical) ---")
    x_opt_mom_num, hist_mom_num = momentum_gradient_descent(
        f=func,
        x_init=x_init,
        learning_rate=learning_rate,
        momentum=momentum,
        max_iter=max_iter,
        use_numerical=True
    )
    iter_mom_num = len(hist_mom_num['f_values']) - 1
    f_final_mom_num = hist_mom_num['f_values'][-1]

    print(f"Iterasi: {iter_mom_num}")
    print(f"x_optimal: {[round(xi, 6) for xi in x_opt_mom_num]}")
    print(f"f(x_optimal): {f_final_mom_num:.6e}")

    # Bandingkan dengan global minimum jika ada
    if func.global_minimum is not None:
        print(f"\n--- Perbandingan dengan Global Minimum ---")
        print(f"Global minimum: {func.global_minimum}")
        print(f"f_min teoritis: {func.f_min}")

        error_vanilla = sum((x_opt_vanilla[i] - func.global_minimum[i])**2
                            for i in range(func.dim)) ** 0.5
        error_mom = sum((x_opt_mom[i] - func.global_minimum[i])**2
                        for i in range(func.dim)) ** 0.5
        print(f"Error (Vanilla): {error_vanilla:.6e}")
        print(f"Error (Momentum): {error_mom:.6e}")

    return {
        'name': name or func.name,
        'vanilla': {'iterations': iter_vanilla, 'f_final': f_final_vanilla, 'x_opt': x_opt_vanilla},
        'momentum': {'iterations': iter_mom, 'f_final': f_final_mom, 'x_opt': x_opt_mom},
        'vanilla_num': {'iterations': iter_num, 'f_final': f_final_num, 'x_opt': x_opt_num},
        'momentum_num': {'iterations': iter_mom_num, 'f_final': f_final_mom_num, 'x_opt': x_opt_mom_num}
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
        momentum=0.9,
        name="Quadratic (convex)"
    ))

    # 2. Rosenbrock 2D (non-convex, classic)
    rosen = Rosenbrock(dim=2)
    results.append(benchmark_function(
        rosen,
        x_init=[-1.0, 1.0],
        learning_rate=0.001,
        momentum=0.9,
        max_iter=10000
    ))

    # 3. Rastrigin 2D (highly multimodal)
    rast = Rastrigin(dim=2)
    results.append(benchmark_function(
        rast,
        x_init=[0.5, 0.5],
        learning_rate=0.01,
        momentum=0.9,
        max_iter=1000
    ))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Function':<20} {'Vanilla':<15} {'Momentum':<15} {'V-Num':<15} {'M-Num':<15}")
    print(f"{'':20} {'(iter/f_val)':<15} {'(iter/f_val)':<15} {'(iter/f_val)':<15} {'(iter/f_val)':<15}")
    print("-" * 70)
    for r in results:
        van = f"{r['vanilla']['iterations']}/{r['vanilla']['f_final']:.1e}"
        mom = f"{r['momentum']['iterations']}/{r['momentum']['f_final']:.1e}"
        van_n = f"{r['vanilla_num']['iterations']}/{r['vanilla_num']['f_final']:.1e}"
        mom_n = f"{r['momentum_num']['iterations']}/{r['momentum_num']['f_final']:.1e}"
        print(f"{r['name']:<20} {van:<15} {mom:<15} {van_n:<15} {mom_n:<15}")

    return results


if __name__ == "__main__":
    run_all_benchmarks()
