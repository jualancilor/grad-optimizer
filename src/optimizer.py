"""
Optimizer - Model matematis gradient descent.
Fokus pada update rule dan iterasi optimisasi.
"""
from src.gradient import numerical_gradient, analytical_gradient


def gradient_descent(f, x_init, learning_rate=0.01, max_iter=1000, tol=1e-6,
                     grad_func=None, use_numerical=False):
    """
    Vanilla Gradient Descent.

    Update rule:
        x_{k+1} = x_k - α * ∇f(x_k)

    Parameters:
    -----------
    f : callable
        Fungsi objektif f: R^n -> R
    x_init : list[float]
        Titik awal x_0.
    learning_rate : float
        Learning rate α.
    max_iter : int
        Jumlah iterasi maksimum.
    tol : float
        Toleransi konvergensi ||x_{k+1} - x_k|| < tol.
    grad_func : callable, optional
        Fungsi gradient analitik. Jika None, gunakan numerical.
    use_numerical : bool
        Paksa gunakan numerical gradient.

    Returns:
    --------
    x : list[float]
        Titik optimal.
    history : dict
        Riwayat optimisasi (x, f_values, gradients).
    """
    x = x_init.copy()
    history = {
        'x': [x.copy()],
        'f_values': [f(x)],
        'gradients': []
    }

    for k in range(max_iter):
        # Hitung gradient (delegasi ke gradient.py)
        if use_numerical or grad_func is None:
            grad = numerical_gradient(f, x)
        else:
            grad = analytical_gradient(grad_func, x)

        history['gradients'].append(grad.copy())

        # Update rule: x_{k+1} = x_k - α * ∇f(x_k)
        x_new = [x[i] - learning_rate * grad[i] for i in range(len(x))]

        history['x'].append(x_new.copy())
        history['f_values'].append(f(x_new))

        # Cek konvergensi
        diff = sum((x_new[i] - x[i]) ** 2 for i in range(len(x))) ** 0.5
        if diff < tol:
            break

        x = x_new

    return x, history
