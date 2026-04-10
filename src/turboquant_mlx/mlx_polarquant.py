import mlx.core as mx
import math

class MLXPolarQuantCompressor:
    def __init__(self, feature_dim: int, theta_bits: int = 3, radius_bits: int = 8, seed: int = 42):
        self.feature_dim = feature_dim
        self.theta_bits = theta_bits
        self.theta_max_idx = (1 << theta_bits) - 1
        self.radius_bits = radius_bits
        self.radius_max_idx = (1 << radius_bits) - 1
        
        assert (feature_dim & (feature_dim - 1)) == 0 and feature_dim > 0, "feature_dim must be power of 2"
        
        # QR init natively via MLX
        mx.random.seed(seed)
        H = mx.random.normal([feature_dim, feature_dim])
        Q, R = mx.linalg.qr(H, stream=mx.cpu)
        d = mx.diag(R)
        self.R = Q * mx.sign(d)
        
    def _quantize_value(self, val: mx.array, v_min: float, v_max: float, max_idx: int) -> mx.array:
        normalized = (val - v_min) / (v_max - v_min)
        normalized = mx.clip(normalized, 0.0, 1.0)
        quantized = mx.round(normalized * max_idx).astype(mx.int16)
        return quantized
        
    def _dequantize_value(self, q_val: mx.array, v_min: float, v_max: float, max_idx: int) -> mx.array:
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
                q_angle = self._quantize_value(angle, -math.pi, math.pi, self.theta_max_idx)
            else:
                q_angle = self._quantize_value(angle, 0.0, math.pi / 2.0, self.theta_max_idx)
                
            angles_list.append(q_angle)
            current = radius
            layer += 1
            
        return angles_list, current

    def _polar_to_cartesian_recursive(self, angles_list: list, radius: mx.array):
        current = radius
        for layer in range(len(angles_list)-1, -1, -1):
            q_angle = angles_list[layer]
            
            if layer == 0:
                angle = self._dequantize_value(q_angle, -math.pi, math.pi, self.theta_max_idx)
            else:
                angle = self._dequantize_value(q_angle, 0.0, math.pi / 2.0, self.theta_max_idx)
                
            even = current * mx.cos(angle)
            odd = current * mx.sin(angle)
            
            b = current.shape[0]
            dim_half = current.shape[1]
            stacked = mx.stack([even, odd], axis=-1)
            current = mx.reshape(stacked, (b, dim_half * 2))
            
        return current

    def compress(self, x: mx.array) -> dict:
        is_single = x.ndim == 1
        if is_single:
            x_b = mx.reshape(x, (1, -1))
        else:
            x_b = x
            
        rotated = mx.matmul(x_b, self.R)
        angles_list, radius = self._cartesian_to_polar_recursive(rotated)
        
        r_max = mx.max(radius).item()
        if r_max == 0:
            r_max = 1e-9
            
        q_radius = self._quantize_value(radius, 0.0, r_max, self.radius_max_idx)
        
        if is_single:
            angles_list = [a[0] for a in angles_list]
            q_radius = q_radius[0, 0]
            
        return {"angles": angles_list, "q_radius": q_radius, "r_max": r_max}

    def decompress(self, compressed: dict) -> mx.array:
        angles_list = compressed["angles"]
        q_radius = compressed["q_radius"]
        r_max = compressed["r_max"]
        
        is_single = (not isinstance(q_radius, list) and getattr(q_radius, 'ndim', -1) == 0) or isinstance(q_radius, (float, int))
        
        if is_single:
            q_radius_b = mx.array([[q_radius]], dtype=mx.int16)
            angles_b = [mx.expand_dims(a, 0) for a in angles_list]
        else:
            q_radius_b = q_radius
            angles_b = angles_list
            
        radius_b = self._dequantize_value(q_radius_b, 0.0, r_max, self.radius_max_idx)
        rotated_approx = self._polar_to_cartesian_recursive(angles_b, radius_b)
        
        original_approx = mx.matmul(rotated_approx, self.R.T)
        
        if is_single:
            original_approx = original_approx[0]
            
        return original_approx
