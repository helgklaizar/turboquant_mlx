import mlx.core as mx
import math

class QJLCompressor:
    def __init__(self, feature_dim: int, num_features: int, seed: int = 42):
        """
        Quantized Johnson-Lindenstrauss compressor natively using MLX (1-bit KV Cache).
        
        :param feature_dim: dimensionality of the source vectors (d)
        :param num_features: number of random features (k) in projection
        :param seed: seed for reproducible projection
        """
        self.feature_dim = feature_dim
        self.num_features = num_features
        mx.random.seed(seed)
        
        # Random projection matrix
        # Elements sampled from N(0, 1)
        self.R = mx.random.normal((feature_dim, num_features))
        
    def compress(self, x: mx.array):
        """
        Compresses a vector or batch of vectors into a 1-bit representation using MLX.
        
        :param x: 1D vector (d,) or batch of vectors (b, d)
        :return: tuple (x_quant, norm_x)
        """
        if x.ndim == 1:
            norm_x = mx.linalg.norm(x)
            # Add batch dim for consistent projected logic
            x_input = x[mx.newaxis]
        else:
            norm_x = mx.linalg.norm(x, axis=1, keepdims=True)
            x_input = x
            
        projected = x_input @ self.R
        x_quant = mx.sign(projected)
        # Handle edge-case where projection is exactly 0
        x_quant = mx.where(x_quant == 0, 1.0, x_quant)
        
        if x.ndim == 1:
            x_quant = x_quant[0]
            
        return x_quant, norm_x
        
    def estimate_dot(self, x_quant: mx.array, norm_x: mx.array, y: mx.array) -> mx.array:
        """
        Asymmetric dot product estimation using MLX.
        
        :param x_quant: quantized feature vector with values {-1, 1}
        :param norm_x: L2 norm or vector of batch norms
        :param y: query vector (float)
        """
        y_proj = y @ self.R
        
        # Matrix multiplication for dot product estimation
        if x_quant.ndim == 2 and y_proj.ndim == 1:
            dot_product = x_quant @ y_proj
        else:
            # For other shapes (e.g. (b, k) and (m, k) -> (b, m))
            dot_product = x_quant @ y_proj.T
            
        scaling_factor = (norm_x / self.num_features) * math.sqrt(math.pi / 2)
        
        # Squeeze logic
        if dot_product.ndim == 1 and norm_x.ndim > 1:
            scaling_factor = mx.squeeze(scaling_factor)
            
        return dot_product * scaling_factor
