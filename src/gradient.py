"""
Modul komputasi gradient.
Berisi: numerical gradient, analytical gradient, gradient checking.
"""


def numerical_gradient(f, x, epsilon=1e-7):
    """
    Hitung gradient secara numerik menggunakan central difference.

    ∂f/∂x_i ≈ (f(x + εe_i) - f(x - εe_i)) / (2ε)

    Parameters:
    -----------
    f : callable
        Fungsi objektif f: R^n -> R
    x : list[float]
        Titik evaluasi.
    epsilon : float
        Step size untuk finite difference.

    Returns:
    --------
    grad : list[float]
        Gradient numerik pada titik x.
    """
    n = len(x)
    grad = [0.0] * n

    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * epsilon)

    return grad


def analytical_gradient(grad_func, x):
    """
    Wrapper untuk gradient analitik.

    Parameters:
    -----------
    grad_func : callable
        Fungsi gradient analitik ∇f: R^n -> R^n
    x : list[float]
        Titik evaluasi.

    Returns:
    --------
    grad : list[float]
        Gradient analitik pada titik x.
    """
    return grad_func(x)


def gradient_check(f, grad_func, x, epsilon=1e-7, tolerance=1e-5):
    """
    Verifikasi gradient analitik dengan gradient numerik.

    Parameters:
    -----------
    f : callable
        Fungsi objektif f: R^n -> R
    grad_func : callable
        Fungsi gradient analitik ∇f: R^n -> R^n
    x : list[float]
        Titik evaluasi.
    epsilon : float
        Step size untuk numerical gradient.
    tolerance : float
        Toleransi error relatif.

    Returns:
    --------
    is_correct : bool
        True jika gradient analitik benar.
    relative_error : float
        Error relatif antara numerical dan analytical.
    details : dict
        Detail perbandingan per komponen.
    """
    grad_numerical = numerical_gradient(f, x, epsilon)
    grad_analytical = analytical_gradient(grad_func, x)

    # Hitung relative error: ||g_num - g_ana|| / (||g_num|| + ||g_ana||)
    diff_norm = _norm([grad_numerical[i] - grad_analytical[i] for i in range(len(x))])
    num_norm = _norm(grad_numerical)
    ana_norm = _norm(grad_analytical)

    denominator = num_norm + ana_norm
    if denominator < 1e-15:
        relative_error = 0.0
    else:
        relative_error = diff_norm / denominator

    is_correct = relative_error < tolerance

    details = {
        'numerical': grad_numerical,
        'analytical': grad_analytical,
        'component_errors': [abs(grad_numerical[i] - grad_analytical[i]) for i in range(len(x))]
    }

    return is_correct, relative_error, details


def _norm(v):
    """Hitung L2 norm dari vektor."""
    return sum(vi ** 2 for vi in v) ** 0.5
