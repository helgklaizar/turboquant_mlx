import mlx.core as mx
import math
from typing import Optional

class PolarQuantCompressor:
    def __init__(self, feature_dim: int, theta_bits: int = 3, radius_bits: int = 8, seed: int = 42):
        """
        PolarQuant Compressor natively using MLX.
        
        :param feature_dim: vector dimensionality (must be a power of two)
        :param theta_bits: bits for angle quantization
        :param radius_bits: bits for radius quantization
        """
        self.feature_dim = feature_dim
        self.theta_bits = theta_bits
        self.radius_bits = radius_bits
        self.max_theta_idx = (1 << theta_bits) - 1
        self.max_radius_idx = (1 << radius_bits) - 1
        
        assert (feature_dim & (feature_dim - 1)) == 0 and feature_dim > 0, "feature_dim must be a power of 2"
        
        mx.random.seed(seed)
        H = mx.random.normal((feature_dim, feature_dim))
        # QR decomposition is only supported on CPU in current MLX version
        Q, R = mx.linalg.qr(H, stream=mx.cpu)
        d = mx.diag(R)
        self.R = Q * mx.where(d >= 0, 1.0, -1.0)
        
    def _quantize_val(self, val: mx.array, v_min: float, v_max: float, max_idx: int) -> mx.array:
        normalized = (val - v_min) / (v_max - v_min)
        normalized = mx.clip(normalized, 0.0, 1.0)
        return mx.round(normalized * max_idx).astype(mx.uint8 if max_idx < 256 else mx.uint16)
        
    def _dequantize_val(self, q_val: mx.array, v_min: float, v_max: float, max_idx: int) -> mx.array:
        normalized = q_val.astype(mx.float32) / max_idx
        return normalized * (v_max - v_min) + v_min

    def _cartesian_to_polar_recursive(self, x: mx.array):
        current = x
        angles_list = []
        layer = 0
        
        while current.shape[1] > 1:
            even = current[:, 0::2]
            odd = current[:, 1::2]
            
            radius = mx.sqrt(even**2 + odd**2)
            angle = mx.arctan2(odd, even)
            
            if layer == 0:
                q_angle = self._quantize_val(angle, -math.pi, math.pi, self.max_theta_idx)
            else:
                q_angle = self._quantize_val(angle, 0.0, math.pi/2, self.max_theta_idx)
                
            angles_list.append(q_angle)
            current = radius
            layer += 1
            
        # Last radius (root) quantization
        r_min, r_max = mx.min(current), mx.max(current)
        q_radius = self._quantize_val(current, r_min, r_max, self.max_radius_idx)
        
        return {
            "angles": angles_list, 
            "q_radius": q_radius, 
            "r_range": (float(r_min), float(r_max))
        }

    def _polar_to_cartesian_recursive(self, compressed: dict):
        angles_list = compressed["angles"]
        q_radius = compressed["q_radius"]
        r_min, r_max = compressed["r_range"]
        
        current = self._dequantize_val(q_radius, r_min, r_max, self.max_radius_idx)
        
        for layer in range(len(angles_list)-1, -1, -1):
            q_angle = angles_list[layer]
            if layer == 0:
                angle = self._dequantize_val(q_angle, -math.pi, math.pi, self.max_theta_idx)
            else:
                angle = self._dequantize_val(q_angle, 0.0, math.pi/2, self.max_theta_idx)
                
            even = current * mx.cos(angle)
            odd = current * mx.sin(angle)
            current = mx.stack([even, odd], axis=-1).reshape(current.shape[0], -1)
            
        return current

    def compress(self, x: mx.array) -> dict:
        if x.ndim == 1: x = x[mx.newaxis]
        rotated = x @ self.R
        return self._cartesian_to_polar_recursive(rotated)

    def decompress(self, compressed: dict) -> mx.array:
        rotated_approx = self._polar_to_cartesian_recursive(compressed)
        return rotated_approx @ self.R.T
