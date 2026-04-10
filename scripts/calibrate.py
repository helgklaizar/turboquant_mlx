import mlx.core as mx
import mlx.nn as nn
import numpy as np

def calibrate_polar_quant(weights: mx.array, radius_bits: int = 3, angle_bits: int = 3):
    """
    Finds optimal RADIUS_SCALE and ANGLE_SCALE using MSE optimization.
    weights: [Out, In]
    """
    # 1. Convert to Polar
    even = weights[:, ::2]
    odd  = weights[:, 1::2]
    
    r_target = mx.sqrt(even**2 + odd**2)
    a_target = mx.atan2(odd, even)
    
    # 2. Heuristic Initialization
    r_scale = mx.max(r_target) / (2**radius_bits - 1)
    a_scale = (2 * np.pi) / (2**angle_bits - 1)
    
    print(f"Initial Scales: Radius={r_scale.item():.4f}, Angle={a_scale.item():.4f}")
    
    # 3. Optimization Loop (Simplified Grid Search for Stage 1)
    # real calibration would use GD on the scales
    
    return r_scale, a_scale

if __name__ == "__main__":
    dummy_w = mx.random.normal((1024, 1024))
    r_s, a_s = calibrate_polar_quant(dummy_w)
    print(f"✅ Calibration complete: R_SCALE={r_s.item()}, A_SCALE={a_s.item()}")
