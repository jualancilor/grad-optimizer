"""
Objective functions for optimization benchmarking.
All functions: f: R^n -> R
"""
from typing import Callable, Tuple
import math

class ObjectiveFunction:
    """Base class for test functions."""
    
    def __init__(self, dim: int):
        self.dim = dim
        self.name: str = "base"
        self.global_minimum: Tuple[float, ...] = None
        self.f_min: float = None
    
    def __call__(self, x: list[float]) -> float:
        """Evaluate f(x)."""
        raise NotImplementedError
    
    def gradient(self, x: list[float]) -> list[float]:
        """Analytical gradient ∇f(x)."""
        raise NotImplementedError


class Quadratic(ObjectiveFunction):
    """
    f(x) = 0.5 * x^T A x - b^T x
    Simplest convex case for sanity check.
    
    Input:  x ∈ R^n
    Output: scalar ∈ R
    Gradient: Ax - b
    """
    def __init__(self, A: list[list[float]], b: list[float]):
        dim = len(b)
        super().__init__(dim)
        self.A = A
        self.b = b
        self.name = "quadratic"
        # Minimum at x* = A^{-1}b (jika A positive definite)
    
    def __call__(self, x: list[float]) -> float:
        # 0.5 * x^T A x - b^T x
        Ax = self._matvec(self.A, x)
        xAx = self._dot(x, Ax)
        bx = self._dot(self.b, x)
        return 0.5 * xAx - bx
    
    def gradient(self, x: list[float]) -> list[float]:
        # ∇f = Ax - b
        Ax = self._matvec(self.A, x)
        return [Ax[i] - self.b[i] for i in range(self.dim)]
    
    @staticmethod
    def _dot(a, b):
        return sum(ai * bi for ai, bi in zip(a, b))
    
    @staticmethod
    def _matvec(A, x):
        return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


class Rosenbrock(ObjectiveFunction):
    """
    f(x) = Σ [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
    
    Classic non-convex test. Global min at (1,1,...,1), f=0
    
    Input:  x ∈ R^n (n ≥ 2)
    Output: scalar ∈ R
    """
    def __init__(self, dim: int = 2):
        super().__init__(dim)
        self.name = "rosenbrock"
        self.global_minimum = tuple([1.0] * dim)
        self.f_min = 0.0
    
    def __call__(self, x: list[float]) -> float:
        total = 0.0
        for i in range(self.dim - 1):
            total += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
        return total
    
    def gradient(self, x: list[float]) -> list[float]:
        grad = [0.0] * self.dim
        for i in range(self.dim - 1):
            grad[i] += -400 * x[i] * (x[i+1] - x[i]**2) - 2 * (1 - x[i])
            grad[i+1] += 200 * (x[i+1] - x[i]**2)
        return grad


class Rastrigin(ObjectiveFunction):
    """
    f(x) = 10n + Σ [x_i^2 - 10*cos(2πx_i)]
    
    Highly multimodal. Global min at origin, f=0
    Tests optimizer's ability to escape local minima.
    """
    def __init__(self, dim: int = 2):
        super().__init__(dim)
        self.name = "rastrigin"
        self.global_minimum = tuple([0.0] * dim)
        self.f_min = 0.0
    
    def __call__(self, x: list[float]) -> float:
        n = self.dim
        return 10 * n + sum(xi**2 - 10 * math.cos(2 * math.pi * xi) for xi in x)
    
    def gradient(self, x: list[float]) -> list[float]:
        return [2 * xi + 20 * math.pi * math.sin(2 * math.pi * xi) for xi in x]