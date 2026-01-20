"""
Test objective functions are correctly implemented.
"""
import pytest
import math
from src.function import Quadratic, Rosenbrock, Rastrigin


class TestQuadratic:
    """Test quadratic function properties."""
    
    def test_value_at_origin(self):
        """f(0) = 0 for A=I, b=0."""
        A = [[1.0, 0.0], [0.0, 1.0]]
        b = [0.0, 0.0]
        f = Quadratic(A, b)
        
        assert abs(f([0.0, 0.0])) < 1e-10
    
    def test_known_value(self):
        """Test against hand-computed value."""
        A = [[2.0, 0.0], [0.0, 2.0]]
        b = [1.0, 1.0]
        f = Quadratic(A, b)
        
        # f([1,1]) = 0.5 * [1,1] @ [[2,0],[0,2]] @ [1,1] - [1,1] @ [1,1]
        #          = 0.5 * 4 - 2 = 0
        x = [1.0, 1.0]
        assert abs(f(x) - 0.0) < 1e-10
    
    def test_positive_definite_convex(self):
        """Positive definite A gives convex f."""
        A = [[4.0, 1.0], [1.0, 2.0]]  # eigenvalues > 0
        b = [0.0, 0.0]
        f = Quadratic(A, b)
        
        # f(0) should be minimum
        assert f([0.0, 0.0]) <= f([1.0, 0.0])
        assert f([0.0, 0.0]) <= f([0.0, 1.0])


class TestRosenbrock:
    """Test Rosenbrock function."""
    
    def test_global_minimum(self):
        """f(1,1,...,1) = 0."""
        for dim in [2, 3, 5]:
            f = Rosenbrock(dim=dim)
            x_star = [1.0] * dim
            
            assert abs(f(x_star)) < 1e-10
    
    def test_positive_away_from_minimum(self):
        """f(x) > 0 for x ≠ (1,...,1)."""
        f = Rosenbrock(dim=2)
        
        assert f([0.0, 0.0]) > 0
        assert f([2.0, 2.0]) > 0
        assert f([-1.0, 1.0]) > 0


class TestRastrigin:
    """Test Rastrigin function."""
    
    def test_global_minimum(self):
        """f(0,0,...,0) = 0."""
        for dim in [2, 5, 10]:
            f = Rastrigin(dim=dim)
            x_star = [0.0] * dim
            
            assert abs(f(x_star)) < 1e-10
    
    def test_local_minima_exist(self):
        """Rastrigin has many local minima at integer points."""
        f = Rastrigin(dim=1)
        
        # f(1) is a local minimum, but f(1) > f(0)
        assert f([1.0]) > f([0.0])
        assert f([1.0]) < f([0.5])  # 0.5 is not at minimum