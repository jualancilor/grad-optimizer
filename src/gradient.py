import numpy as np


def gradient_descent(gradient_func, x_init, learning_rate=0.01, max_iter=1000, tol=1e-6):
    """
    Vanilla Gradient Descent optimization.

    Parameters:
    -----------
    gradient_func : callable
        Fungsi yang menghitung gradient pada titik x.
    x_init : np.ndarray
        Titik awal optimisasi.
    learning_rate : float
        Learning rate (alpha).
    max_iter : int
        Jumlah iterasi maksimum.
    tol : float
        Toleransi untuk konvergensi.

    Returns:
    --------
    x : np.ndarray
        Titik optimal yang ditemukan.
    history : list
        Riwayat titik selama optimisasi.
    """
    x = np.array(x_init, dtype=float)
    history = [x.copy()]

    for i in range(max_iter):
        grad = gradient_func(x)
        x_new = x - learning_rate * grad
        history.append(x_new.copy())

        # Cek konvergensi
        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new

    return x, history


