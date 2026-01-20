"""
Unit tests for polynomial_solver.py

Run with:
    python -m pytest newton/_src/solvers/vbd/test_polynomial_solver.py -v
Or directly:
    python newton/_src/solvers/vbd/test_polynomial_solver.py
"""

import numpy as np
import warp as wp

# Import the functions we want to test
from polynomial_solver import (
    quadratic_first_root_bounded,
    bisect_cubic,
    cubic_roots_bounded,
)


def test_quadratic_basic():
    """Test quadratic solver with simple cases."""
    wp.init()
    
    @wp.kernel
    def test_kernel(results: wp.array(dtype=float)):
        # Test 1: x^2 - 1 = 0, roots at x = -1, 1
        # coef0 + coef1*x + coef2*x^2 = -1 + 0*x + 1*x^2
        r1 = quadratic_first_root_bounded(-1.0, 0.0, 1.0, 0.0, 2.0)
        results[0] = r1  # Should be 1.0
        
        # Test 2: x^2 - 4 = 0, roots at x = -2, 2
        r2 = quadratic_first_root_bounded(-4.0, 0.0, 1.0, 0.0, 3.0)
        results[1] = r2  # Should be 2.0
        
        # Test 3: x^2 + 1 = 0, no real roots
        r3 = quadratic_first_root_bounded(1.0, 0.0, 1.0, -10.0, 10.0)
        results[2] = r3  # Should be -1.0 (no root)
        
        # Test 4: (x-0.5)(x-0.8) = x^2 - 1.3x + 0.4, roots at 0.5, 0.8
        r4 = quadratic_first_root_bounded(0.4, -1.3, 1.0, 0.0, 1.0)
        results[3] = r4  # Should be ~0.5
        
        # Test 5: Linear case (coef2 = 0): 2x - 1 = 0, root at 0.5
        r5 = quadratic_first_root_bounded(-1.0, 2.0, 0.0, 0.0, 1.0)
        results[4] = r5  # Should be 0.5
    
    results = wp.zeros(5, dtype=float, device="cpu")
    wp.launch(test_kernel, dim=1, inputs=[results], device="cpu")
    wp.synchronize()
    
    r = results.numpy()
    print("\n=== Quadratic Solver Tests ===")
    print(f"Test 1 (x²-1=0 in [0,2]): root = {r[0]:.6f}, expected = 1.0, pass = {np.isclose(r[0], 1.0, atol=1e-6)}")
    print(f"Test 2 (x²-4=0 in [0,3]): root = {r[1]:.6f}, expected = 2.0, pass = {np.isclose(r[1], 2.0, atol=1e-6)}")
    print(f"Test 3 (x²+1=0, no root): root = {r[2]:.6f}, expected = -1.0, pass = {r[2] < 0}")
    print(f"Test 4 ((x-0.5)(x-0.8)=0): root = {r[3]:.6f}, expected ~ 0.5, pass = {np.isclose(r[3], 0.5, atol=1e-6)}")
    print(f"Test 5 (2x-1=0, linear): root = {r[4]:.6f}, expected = 0.5, pass = {np.isclose(r[4], 0.5, atol=1e-6)}")
    
    assert np.isclose(r[0], 1.0, atol=1e-6), f"Test 1 failed: {r[0]}"
    assert np.isclose(r[1], 2.0, atol=1e-6), f"Test 2 failed: {r[1]}"
    assert r[2] < 0, f"Test 3 failed: {r[2]}"
    assert np.isclose(r[3], 0.5, atol=1e-6), f"Test 4 failed: {r[3]}"
    assert np.isclose(r[4], 0.5, atol=1e-6), f"Test 5 failed: {r[4]}"
    print("All quadratic tests PASSED [OK]")


def test_cubic_basic():
    """Test cubic solver with simple cases."""
    wp.init()
    
    @wp.kernel
    def test_kernel(results: wp.array(dtype=float)):
        # Test 1: x^3 - 1 = 0, root at x = 1
        r1 = cubic_roots_bounded(-1.0, 0.0, 0.0, 1.0, 0.0, 2.0)
        results[0] = r1  # Should be 1.0
        
        # Test 2: x^3 - 8 = 0, root at x = 2
        r2 = cubic_roots_bounded(-8.0, 0.0, 0.0, 1.0, 0.0, 3.0)
        results[1] = r2  # Should be 2.0
        
        # Test 3: (x-0.3)(x-0.6)(x-0.9) = x^3 - 1.8x^2 + 0.99x - 0.162
        # Roots at 0.3, 0.6, 0.9
        r3 = cubic_roots_bounded(-0.162, 0.99, -1.8, 1.0, 0.0, 1.0)
        results[2] = r3  # Should be ~0.3 (first root)
        
        # Test 4: x^3 + x = 0, only root at x = 0
        r4 = cubic_roots_bounded(0.0, 1.0, 0.0, 1.0, -1.0, 1.0)
        results[3] = r4  # Should be 0.0
        
        # Test 5: No root in range - x^3 - 1 = 0, but search in [2, 3]
        r5 = cubic_roots_bounded(-1.0, 0.0, 0.0, 1.0, 2.0, 3.0)
        results[4] = r5  # Should be -1.0 (no root)
        
        # Test 6: CCD-like polynomial with root at t=0.5
        # Construct: (t - 0.5) * (t^2 + 1) = t^3 - 0.5t^2 + t - 0.5
        r6 = cubic_roots_bounded(-0.5, 1.0, -0.5, 1.0, 0.0, 1.0)
        results[5] = r6  # Should be 0.5
    
    results = wp.zeros(6, dtype=float, device="cpu")
    wp.launch(test_kernel, dim=1, inputs=[results], device="cpu")
    wp.synchronize()
    
    r = results.numpy()
    print("\n=== Cubic Solver Tests ===")
    print(f"Test 1 (x³-1=0 in [0,2]): root = {r[0]:.6f}, expected = 1.0, pass = {np.isclose(r[0], 1.0, atol=1e-5)}")
    print(f"Test 2 (x³-8=0 in [0,3]): root = {r[1]:.6f}, expected = 2.0, pass = {np.isclose(r[1], 2.0, atol=1e-5)}")
    print(f"Test 3 (three roots, first): root = {r[2]:.6f}, expected ~ 0.3, pass = {np.isclose(r[2], 0.3, atol=1e-5)}")
    print(f"Test 4 (x³+x=0): root = {r[3]:.6f}, expected = 0.0, pass = {np.isclose(r[3], 0.0, atol=1e-5)}")
    print(f"Test 5 (no root in [2,3]): root = {r[4]:.6f}, expected = -1.0, pass = {r[4] < 0}")
    print(f"Test 6 (CCD-like, t=0.5): root = {r[5]:.6f}, expected = 0.5, pass = {np.isclose(r[5], 0.5, atol=1e-5)}")
    
    assert np.isclose(r[0], 1.0, atol=1e-5), f"Test 1 failed: {r[0]}"
    assert np.isclose(r[1], 2.0, atol=1e-5), f"Test 2 failed: {r[1]}"
    assert np.isclose(r[2], 0.3, atol=1e-5), f"Test 3 failed: {r[2]}"
    assert np.isclose(r[3], 0.0, atol=1e-5), f"Test 4 failed: {r[3]}"
    assert r[4] < 0, f"Test 5 failed: {r[4]}"
    assert np.isclose(r[5], 0.5, atol=1e-5), f"Test 6 failed: {r[5]}"
    print("All cubic tests PASSED [OK]")


def test_cubic_random():
    """Test cubic solver with random polynomials."""
    wp.init()
    
    np.random.seed(42)
    num_tests = 10000
    
    # Generate random roots in [0, 1] with minimum separation
    # (roots too close together cause numerical issues - expected behavior)
    roots = []
    min_sep = 0.1  # Minimum separation between roots
    for _ in range(num_tests):
        while True:
            r = np.random.uniform(0.1, 0.9, 3)
            r.sort()
            if r[1] - r[0] > min_sep and r[2] - r[1] > min_sep:
                roots.append(r)
                break
    roots = np.array(roots)
    
    # Compute polynomial coefficients from roots
    # (x - r1)(x - r2)(x - r3) = x^3 - (r1+r2+r3)x^2 + (r1*r2+r1*r3+r2*r3)x - r1*r2*r3
    coef0 = -roots[:, 0] * roots[:, 1] * roots[:, 2]
    coef1 = roots[:, 0] * roots[:, 1] + roots[:, 0] * roots[:, 2] + roots[:, 1] * roots[:, 2]
    coef2 = -(roots[:, 0] + roots[:, 1] + roots[:, 2])
    coef3 = np.ones(num_tests)
    
    @wp.kernel
    def test_kernel(
        c0: wp.array(dtype=float),
        c1: wp.array(dtype=float),
        c2: wp.array(dtype=float),
        c3: wp.array(dtype=float),
        results: wp.array(dtype=float),
    ):
        i = wp.tid()
        results[i] = cubic_roots_bounded(c0[i], c1[i], c2[i], c3[i], 0.0, 1.0)
    
    c0_wp = wp.array(coef0.astype(np.float32), dtype=float, device="cpu")
    c1_wp = wp.array(coef1.astype(np.float32), dtype=float, device="cpu")
    c2_wp = wp.array(coef2.astype(np.float32), dtype=float, device="cpu")
    c3_wp = wp.array(coef3.astype(np.float32), dtype=float, device="cpu")
    results = wp.zeros(num_tests, dtype=float, device="cpu")
    
    wp.launch(test_kernel, dim=num_tests, inputs=[c0_wp, c1_wp, c2_wp, c3_wp, results], device="cpu")
    wp.synchronize()
    
    r = results.numpy()
    expected = roots[:, 0]  # First (smallest) root
    
    errors = np.abs(r - expected)
    max_error = np.max(errors)
    mean_error = np.mean(errors)
    
    print(f"\n=== Random Cubic Tests ({num_tests} polynomials) ===")
    print(f"Max error: {max_error:.2e}")
    print(f"Mean error: {mean_error:.2e}")
    print(f"All within 1e-5: {np.all(errors < 1e-5)}")
    
    # Show a few examples
    print("\nSample results (first 5):")
    for i in range(5):
        print(f"  Roots: [{roots[i,0]:.4f}, {roots[i,1]:.4f}, {roots[i,2]:.4f}], found: {r[i]:.6f}, error: {errors[i]:.2e}")
    
    assert np.all(errors < 1e-4), f"Max error too large: {max_error}"
    print(f"All {num_tests} random cubic tests PASSED [OK]")


def test_degenerate_cases():
    """Test edge cases and degenerate polynomials."""
    wp.init()
    
    @wp.kernel
    def test_kernel(results: wp.array(dtype=float)):
        # Test 1: Cubic with zero leading coefficient (actually quadratic)
        # 0*x^3 + x^2 - 1 = 0, roots at +/-1
        r1 = cubic_roots_bounded(-1.0, 0.0, 1.0, 0.0, 0.0, 2.0)
        results[0] = r1  # Should be 1.0
        
        # Test 2: Very small coefficients
        r2 = cubic_roots_bounded(-1e-8, 0.0, 0.0, 1e-8, 0.0, 2.0)
        results[1] = r2  # Should be ~1.0
        
        # Test 3: Roots at 0.1, 0.5, 0.9 (well separated)
        # (x-0.1)(x-0.5)(x-0.9) = x^3 - 1.5x^2 + 0.59x - 0.045
        r3 = cubic_roots_bounded(-0.045, 0.59, -1.5, 1.0, 0.0, 1.0)
        results[2] = r3  # Should be ~0.1
        
        # Test 4: Root very close to 0
        # (x - 0.001) * (x^2 + 1)
        r4 = cubic_roots_bounded(-0.001, 1.0, -0.001, 1.0, 0.0, 1.0)
        results[3] = r4  # Should be ~0.001
        
        # Test 5: Single root case (other two are complex)
        # x^3 + x - 2 = 0 has root at x=1
        r5 = cubic_roots_bounded(-2.0, 1.0, 0.0, 1.0, 0.0, 2.0)
        results[4] = r5  # Should be 1.0
    
    results = wp.zeros(5, dtype=float, device="cpu")
    wp.launch(test_kernel, dim=1, inputs=[results], device="cpu")
    wp.synchronize()
    
    r = results.numpy()
    print("\n=== Degenerate Case Tests ===")
    print(f"Test 1 (zero cubic coef): root = {r[0]:.6f}, expected = 1.0, pass = {np.isclose(r[0], 1.0, atol=1e-5)}")
    print(f"Test 2 (tiny coefficients): root = {r[1]:.6f}, expected ~ 1.0, pass = {np.isclose(r[1], 1.0, atol=1e-4)}")
    print(f"Test 3 (three roots): root = {r[2]:.6f}, expected ~ 0.1, pass = {np.isclose(r[2], 0.1, atol=1e-5)}")
    print(f"Test 4 (root near 0): root = {r[3]:.6f}, expected ~ 0.001, pass = {np.isclose(r[3], 0.001, atol=1e-4)}")
    print(f"Test 5 (single real root): root = {r[4]:.6f}, expected = 1.0, pass = {np.isclose(r[4], 1.0, atol=1e-5)}")
    
    assert np.isclose(r[0], 1.0, atol=1e-5), f"Test 1 failed: {r[0]}"
    assert np.isclose(r[1], 1.0, atol=1e-4), f"Test 2 failed: {r[1]}"
    assert np.isclose(r[2], 0.1, atol=1e-5), f"Test 3 failed: {r[2]}"
    assert np.isclose(r[3], 0.001, atol=1e-4), f"Test 4 failed: {r[3]}"
    assert np.isclose(r[4], 1.0, atol=1e-5), f"Test 5 failed: {r[4]}"
    print("All degenerate case tests PASSED [OK]")


if __name__ == "__main__":
    print("=" * 60)
    print("Polynomial Solver Unit Tests")
    print("=" * 60)
    
    test_quadratic_basic()
    test_cubic_basic()
    test_cubic_random()
    test_degenerate_cases()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! [OK]")
    print("=" * 60)
