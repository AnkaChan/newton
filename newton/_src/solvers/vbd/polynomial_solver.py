# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Polynomial root finding for Warp kernels.

Based on cyPolynomial.h by Cem Yuksel:
http://www.cemyuksel.com/?x=polynomials

Reference:
Cem Yuksel. 2022. High-Performance Polynomial Root Finding for Graphics.
Proc. ACM Comput. Graph. Interact. Tech. 5, 3, Article 7 (July 2022), 15 pages.
"""

import warp as wp


@wp.func
def quadratic_first_root_bounded(
    coef0: float, coef1: float, coef2: float,
    x_min: float, x_max: float
) -> float:
    """
    Find first root of quadratic polynomial in [x_min, x_max].
    
    Polynomial: coef0 + coef1*x + coef2*x^2
    
    Returns:
        First root in range, or -1.0 if none found.
    """
    if wp.abs(coef2) < 1e-12:
        # Linear case: coef0 + coef1*x = 0
        if wp.abs(coef1) < 1e-12:
            return -1.0
        root = -coef0 / coef1
        if root >= x_min and root <= x_max:
            return root
        return -1.0
    
    # Quadratic formula with numerical stability
    delta = coef1 * coef1 - 4.0 * coef0 * coef2
    if delta < 0.0:
        return -1.0
    
    d = wp.sqrt(delta)
    
    # Use stable formula: q = -0.5 * (b + sign(b) * sqrt(delta))
    q = -0.5 * (coef1 + wp.sign(coef1) * d)
    if wp.abs(coef1) < 1e-12:
        q = -0.5 * d
    
    r0 = q / coef2
    r1 = coef0 / q if wp.abs(q) > 1e-12 else 0.0
    
    root0 = wp.min(r0, r1)
    root1 = wp.max(r0, r1)
    
    if root0 >= x_min and root0 <= x_max:
        return root0
    if root1 >= x_min and root1 <= x_max:
        return root1
    return -1.0


@wp.func
def bisect_cubic(
    coef0: float, coef1: float, coef2: float, coef3: float,
    x0: float, x1: float, y0: float, y1: float
) -> float:
    """
    Find root of cubic in [x0, x1] using Newton iterations with bisection fallback.
    
    Precondition: sign(y0) != sign(y1), guaranteeing a root exists in [x0, x1].
    
    Polynomial: coef0 + coef1*x + coef2*x^2 + coef3*x^3
    """
    xr = (x0 + x1) * 0.5
    
    # Newton iterations with bisection fallback
    for _ in range(32):
        yr = coef0 + coef1 * xr + coef2 * xr * xr + coef3 * xr * xr * xr
        
        # Update bounds based on intermediate value theorem
        if (y0 < 0.0) != (yr < 0.0):
            x1 = xr
            y1 = yr
        else:
            x0 = xr
            y0 = yr
        
        # Try Newton step
        dyr = coef1 + 2.0 * coef2 * xr + 3.0 * coef3 * xr * xr
        if wp.abs(dyr) > 1e-12:
            xn = xr - yr / dyr
            # Accept Newton step only if it stays within bounds
            if xn > x0 and xn < x1:
                xr = xn
            else:
                # Fallback to bisection
                xr = (x0 + x1) * 0.5
        else:
            # Derivative too small, use bisection
            xr = (x0 + x1) * 0.5
        
        # Convergence check
        if x1 - x0 < 1e-10:
            break
    
    return xr


@wp.func
def cubic_roots_bounded(
    coef0: float, coef1: float, coef2: float, coef3: float,
    x_min: float, x_max: float
) -> float:
    """
    Find the first root of cubic polynomial in [x_min, x_max].
    
    Polynomial: coef0 + coef1*x + coef2*x^2 + coef3*x^3
    
    Uses the derivative to find critical points and isolate root intervals,
    then applies Newton iterations with bisection for robust convergence.
    
    Returns:
        First root in range, or -1.0 if none found.
    """
    # Handle degenerate case where leading coefficient is zero
    if wp.abs(coef3) < 1e-12:
        return quadratic_first_root_bounded(coef0, coef1, coef2, x_min, x_max)
    
    # Evaluate polynomial at bounds
    y0 = coef0 + coef1 * x_min + coef2 * x_min * x_min + coef3 * x_min * x_min * x_min
    y1 = coef0 + coef1 * x_max + coef2 * x_max * x_max + coef3 * x_max * x_max * x_max
    
    # Derivative: coef1 + 2*coef2*x + 3*coef3*x^2
    # Critical points where derivative = 0
    a = coef3 * 3.0
    b_2 = coef2  # b/2 for numerical stability
    c = coef1
    
    # Discriminant of derivative (divided by 4)
    delta_4 = b_2 * b_2 - a * c
    
    if delta_4 > 0.0:
        # Two distinct critical points - cubic has local min and max
        d_2 = wp.sqrt(delta_4)
        
        # Stable quadratic formula for derivative roots
        q = -b_2 - wp.sign(b_2) * d_2
        if wp.abs(b_2) < 1e-12:
            q = -d_2
        
        rv0 = q / a
        rv1 = c / q if wp.abs(q) > 1e-12 else 0.0
        
        xa = wp.min(rv0, rv1)  # First critical point
        xb = wp.max(rv0, rv1)  # Second critical point
        
        # Check each monotonic interval for roots
        if xa > x_min and xa < x_max:
            ya = coef0 + coef1 * xa + coef2 * xa * xa + coef3 * xa * xa * xa
            
            # Interval [x_min, xa]
            if (y0 < 0.0) != (ya < 0.0):
                return bisect_cubic(coef0, coef1, coef2, coef3, x_min, xa, y0, ya)
            
            # Check remaining intervals
            if xb > x_min and xb < x_max:
                yb = coef0 + coef1 * xb + coef2 * xb * xb + coef3 * xb * xb * xb
                
                # Interval [xa, xb]
                if (ya < 0.0) != (yb < 0.0):
                    return bisect_cubic(coef0, coef1, coef2, coef3, xa, xb, ya, yb)
                
                # Interval [xb, x_max]
                if (yb < 0.0) != (y1 < 0.0):
                    return bisect_cubic(coef0, coef1, coef2, coef3, xb, x_max, yb, y1)
            else:
                # Interval [xa, x_max]
                if (ya < 0.0) != (y1 < 0.0):
                    return bisect_cubic(coef0, coef1, coef2, coef3, xa, x_max, ya, y1)
                    
        elif xb > x_min and xb < x_max:
            yb = coef0 + coef1 * xb + coef2 * xb * xb + coef3 * xb * xb * xb
            
            # Interval [x_min, xb]
            if (y0 < 0.0) != (yb < 0.0):
                return bisect_cubic(coef0, coef1, coef2, coef3, x_min, xb, y0, yb)
            
            # Interval [xb, x_max]
            if (yb < 0.0) != (y1 < 0.0):
                return bisect_cubic(coef0, coef1, coef2, coef3, xb, x_max, yb, y1)
        else:
            # No critical points in range - monotonic
            if (y0 < 0.0) != (y1 < 0.0):
                return bisect_cubic(coef0, coef1, coef2, coef3, x_min, x_max, y0, y1)
    else:
        # No real critical points or repeated - monotonic in [x_min, x_max]
        if (y0 < 0.0) != (y1 < 0.0):
            return bisect_cubic(coef0, coef1, coef2, coef3, x_min, x_max, y0, y1)
    
    return -1.0
